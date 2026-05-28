from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard


class MCTSNode:
    """Search tree node with AlphaZero statistics."""

    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: dict[int, MCTSNode] = {}

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search."""

    def __init__(
        self,
        model: nn.Module,
        c_puct: float = 1.5,
        n_simulations: int = 400,
        device: str = "cpu",
        use_amp: bool = True,
        cache_size: int = 20_000,
        leaf_batch_size: int = 8,
        dirichlet_alpha: float = 0.10,
        dirichlet_frac: float = 0.25,
        fpu_reduction: float = 0.25,
        virtual_loss: float = 1.0,
        prior_uniform_mix: float = 0.0,
        forced_playout_k: float = 0.0,
        policy_target_prune_forced: bool = False,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.use_amp = use_amp and self.device.type == "cuda"
        self.cache_size = max(0, int(cache_size))
        self.leaf_batch_size = max(1, int(leaf_batch_size))
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_frac = float(dirichlet_frac)
        self.fpu_reduction = float(fpu_reduction)
        self.virtual_loss = float(virtual_loss)
        # v15.2: mezcla de prior con uniforme. prior_eff = (1-a)*p + a*(1/N).
        # Evita que un policy sobreconfiado colapse el search a 1-2 acciones.
        self.prior_uniform_mix = max(0.0, min(1.0, float(prior_uniform_mix)))
        # v15 final: Forced Playouts (KataGo). En root, cada hijo con prior > 0
        # recibe un minimo de visitas n_forced = k * prior * sqrt(N_root). Si
        # alguno esta por debajo, se prioriza antes que el PUCT normal.
        self.forced_playout_k = max(0.0, float(forced_playout_k))
        self.policy_target_prune_forced = bool(policy_target_prune_forced)
        self._inference_cache: OrderedDict[
            bytes,
            tuple[np.ndarray, np.ndarray, float],
        ] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _sample_tied_index(candidate_indices: np.ndarray) -> int:
        if candidate_indices.size == 1:
            return int(candidate_indices[0])
        picked = int(np.random.randint(0, candidate_indices.size))
        return int(candidate_indices[picked])

    def run(
        self,
        board: AtaxxBoard,
        add_dirichlet_noise: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Run MCTS from `board` and return visit-based policy probabilities.
        """
        probs, _ = self.run_with_root(
            board=board,
            root=None,
            add_dirichlet_noise=add_dirichlet_noise,
            temperature=temperature,
        )
        return probs

    def run_with_root(
        self,
        board: AtaxxBoard,
        root: MCTSNode | None,
        add_dirichlet_noise: bool = False,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, MCTSNode | None]:
        """
        Run MCTS from `board` starting at an optional cached root.
        Returns (visit_probs, root_after_search).
        """
        self._cache_hits = 0
        self._cache_misses = 0
        probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        if board.is_game_over():
            return probs, None

        if root is None:
            root = MCTSNode(prior=1.0)
            self._expand(root, board)
        elif not root.children:
            self._expand(root, board)

        if add_dirichlet_noise and root.children:
            self._add_dirichlet_noise(root, alpha=self.dirichlet_alpha, frac=self.dirichlet_frac)

        sims_done = 0
        while sims_done < self.n_simulations:
            batch_budget = min(self.leaf_batch_size, self.n_simulations - sims_done)
            pending: list[tuple[MCTSNode, AtaxxBoard, list[MCTSNode]]] = []

            for _ in range(batch_budget):
                node = root
                scratch_board = board.copy()
                search_path = [node]

                is_root_descent = True
                while node.children:
                    action_idx, node = self._select_child(node, is_root=is_root_descent)
                    is_root_descent = False
                    move = ACTION_SPACE.decode(action_idx)
                    scratch_board.step(move)
                    search_path.append(node)

                if scratch_board.is_game_over():
                    value = self._terminal_value_for_current_player(scratch_board)
                    self._backpropagate(search_path, value)
                else:
                    # Virtual loss: penalize this branch temporarily so subsequent
                    # sims in the same batch don't re-select the same leaf. Reverted
                    # after expand_batch when the real value is backpropagated.
                    if self.virtual_loss > 0.0 and len(search_path) > 1:
                        self._apply_virtual_loss(search_path)
                    pending.append((node, scratch_board, search_path))
                sims_done += 1

            if pending:
                leaves = [(node, leaf_board) for node, leaf_board, _ in pending]
                values = self._expand_batch(leaves)
                for (_, _, search_path), value in zip(pending, values, strict=True):
                    if self.virtual_loss > 0.0 and len(search_path) > 1:
                        self._revert_virtual_loss(search_path)
                    self._backpropagate(search_path, value)

        return self._get_action_probs(root, temperature), root

    def advance_root(self, root: MCTSNode | None, action_idx: int) -> MCTSNode | None:
        """Advance cached root after applying `action_idx` on the real board."""
        if root is None:
            return None
        return root.children.get(action_idx)

    @staticmethod
    def top_n_actions(
        root: MCTSNode | None,
        n: int = 3,
    ) -> list[tuple[int, int, float, float]]:
        """Top-N children of `root` ordered by visit_count desc.

        Returns list of (action_idx, visits, value, prior). Used by HUD.
        """
        if root is None or not root.children:
            return []
        ranked = sorted(
            root.children.items(),
            key=lambda item: item[1].visit_count,
            reverse=True,
        )
        return [
            (int(action_idx), int(child.visit_count), float(child.value()), float(child.prior))
            for action_idx, child in ranked[: max(0, int(n))]
        ]

    def _populate_children(
        self,
        node: MCTSNode,
        legal_action_indices: np.ndarray,
        legal_priors: np.ndarray,
    ) -> None:
        if legal_action_indices.size == 0:
            return
        prior_sum = float(np.sum(legal_priors))
        if prior_sum <= 0.0:
            uniform_prior = 1.0 / float(legal_action_indices.size)
            for action_idx in legal_action_indices:
                node.children[int(action_idx)] = MCTSNode(prior=uniform_prior)
            return
        normalized = legal_priors / prior_sum
        if self.prior_uniform_mix > 0.0:
            n = float(legal_action_indices.size)
            normalized = (1.0 - self.prior_uniform_mix) * normalized + (
                self.prior_uniform_mix / n
            )
        for action_idx, prior in zip(legal_action_indices, normalized, strict=True):
            node.children[int(action_idx)] = MCTSNode(prior=float(prior))

    @staticmethod
    def _observation_cache_key(observation: np.ndarray) -> bytes:
        # Cache entries must be keyed by the exact network input. Using only the
        # piece grid aliases states that differ in auxiliary channels such as the
        # half-move counter, which corrupts search with stale priors/values.
        return observation.tobytes()

    def _expand_batch(self, leaves: list[tuple[MCTSNode, AtaxxBoard]]) -> list[float]:
        results: list[float] = [0.0] * len(leaves)
        to_infer: list[tuple[int, MCTSNode, np.ndarray, bytes, np.ndarray]] = []

        for idx, (node, board) in enumerate(leaves):
            obs = board.get_observation()
            cache_key = self._observation_cache_key(obs)
            cached = self._inference_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                legal_action_indices, legal_priors, value = cached
                self._inference_cache.move_to_end(cache_key)
                self._populate_children(node, legal_action_indices, legal_priors)
                results[idx] = value
                continue
            self._cache_misses += 1

            valid_moves = board.get_valid_moves()
            if len(valid_moves) == 0:
                legal_action_indices = np.array([ACTION_SPACE.pass_index], dtype=np.int64)
            else:
                legal_action_indices = np.fromiter(
                    (ACTION_SPACE.encode(move) for move in valid_moves),
                    dtype=np.int64,
                    count=len(valid_moves),
                )
            to_infer.append((idx, node, legal_action_indices, cache_key, obs))

        if not to_infer:
            return results

        states = torch.stack(
            [torch.from_numpy(obs) for _, _, _, _, obs in to_infer],
            dim=0,
        ).to(self.device)
        action_mask = torch.zeros(
            (len(to_infer), ACTION_SPACE.num_actions),
            device=self.device,
            dtype=states.dtype,
        )
        for batch_idx, (_, _, legal_action_indices, _, _) in enumerate(to_infer):
            legal_idx_tensor = torch.from_numpy(legal_action_indices).to(self.device)
            action_mask[batch_idx, legal_idx_tensor] = 1.0

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp_ctx:
            policy_logits, value_tensor = self.model(states, action_mask=action_mask)

        values_np = value_tensor.squeeze(1).detach().float().cpu().numpy()
        for batch_idx, (idx, node, legal_action_indices, cache_key, _) in enumerate(to_infer):
            legal_idx_tensor = torch.from_numpy(legal_action_indices).to(self.device)
            legal_logits = policy_logits[batch_idx].index_select(0, legal_idx_tensor)
            legal_priors = torch.softmax(legal_logits, dim=0).float().cpu().numpy()
            value = float(values_np[batch_idx])
            self._populate_children(node, legal_action_indices, legal_priors)
            results[idx] = value
            if self.cache_size > 0:
                self._inference_cache[cache_key] = (
                    legal_action_indices.copy(),
                    legal_priors.copy(),
                    value,
                )
                if len(self._inference_cache) > self.cache_size:
                    self._inference_cache.popitem(last=False)

        return results

    def _add_dirichlet_noise(self, node: MCTSNode, alpha: float, frac: float) -> None:
        actions = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for idx, action_idx in enumerate(actions):
            child = node.children[action_idx]
            child.prior = (1.0 - frac) * child.prior + frac * float(noise[idx])

    def _select_child(
        self,
        node: MCTSNode,
        is_root: bool = False,
    ) -> tuple[int, MCTSNode]:
        # FPTP forced playout: en root, si algun hijo con prior > 0 esta debajo
        # de su cuota de visitas forzadas (k * prior * sqrt(N_root)), lo elegimos
        # antes que PUCT normal. Solo aplica en root y solo si k > 0.
        if is_root and self.forced_playout_k > 0.0 and node.visit_count > 0:
            sqrt_root = math.sqrt(float(node.visit_count))
            best_deficit = 0.0
            forced_action: int | None = None
            forced_child: MCTSNode | None = None
            for action_idx, child in node.children.items():
                if child.prior <= 0.0:
                    continue
                n_forced = math.ceil(self.forced_playout_k * child.prior * sqrt_root)
                deficit = float(n_forced - child.visit_count)
                if deficit > best_deficit:
                    best_deficit = deficit
                    forced_action = action_idx
                    forced_child = child
            if forced_child is not None and forced_action is not None:
                return forced_action, forced_child

        best_score = -float("inf")
        tied_actions: list[int] = []
        tied_children: list[MCTSNode] = []
        sqrt_parent = math.sqrt(node.visit_count + 1)
        # FPU: non-visited siblings get parent_value - fpu_reduction*sqrt(visited_prior_mass).
        # Without FPU, root with a flat policy gives Q=0 to every child and the search
        # collapses into ties until value backups differentiate (very slow early training).
        parent_value = node.value()
        visited_prior_sum = 0.0
        for child in node.children.values():
            if child.visit_count > 0:
                visited_prior_sum += child.prior
        fpu_offset = self.fpu_reduction * math.sqrt(visited_prior_sum)
        fpu_q = parent_value - fpu_offset

        for action_idx, child in node.children.items():
            # child.value() is from child-player perspective; negate for parent.
            # Non-visited children use FPU (parent_value reduced by visited prior mass).
            q_value = fpu_q if child.visit_count == 0 else -child.value()
            u_value = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_value + u_value
            # Early training often produces flat priors/value estimates. If we always
            # keep the first child on exact ties, search collapses into one opening.
            if score > (best_score + 1e-12):
                best_score = score
                tied_actions = [action_idx]
                tied_children = [child]
                continue
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12):
                tied_actions.append(action_idx)
                tied_children.append(child)

        if len(tied_children) == 0:
            raise RuntimeError("No child selected from a non-empty node.")
        picked = self._sample_tied_index(np.arange(len(tied_children), dtype=np.int64))
        return tied_actions[picked], tied_children[picked]

    def _expand(self, node: MCTSNode, board: AtaxxBoard) -> float:
        """
        Expand a leaf node and return value in current-player perspective.
        """
        return self._expand_batch([(node, board)])[0]

    def _terminal_value_for_current_player(self, board: AtaxxBoard) -> float:
        winner = board.get_result()
        if winner == 0:
            return 0.0
        return 1.0 if winner == board.current_player else -1.0

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _apply_virtual_loss(self, path: list[MCTSNode]) -> None:
        # Increment visits + add virtual_loss to value_sum so child.value() looks
        # better-for-itself (i.e. worse-for-parent via the q_parent = -child.value()
        # flip in _select_child). Skip the root: its value() is informational only.
        for node in path[1:]:
            node.visit_count += 1
            node.value_sum += self.virtual_loss

    def _revert_virtual_loss(self, path: list[MCTSNode]) -> None:
        for node in path[1:]:
            node.visit_count -= 1
            node.value_sum -= self.virtual_loss

    def _get_action_probs(self, root: MCTSNode, temperature: float) -> np.ndarray:
        probs = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
        if not root.children:
            return probs

        actions = np.array(list(root.children.keys()), dtype=np.int64)
        visit_counts = np.array(
            [root.children[action].visit_count for action in actions],
            dtype=np.float32,
        )

        # FPTP policy target pruning: restamos las visitas forzadas del visit
        # count antes de calcular el policy target — esas visitas fueron por
        # cuota, no por search organico, asi que sesgan el target. La accion
        # mas visitada NO se podia (se queda como peak del policy).
        if (
            self.policy_target_prune_forced
            and self.forced_playout_k > 0.0
            and root.visit_count > 0
        ):
            sqrt_root = math.sqrt(float(root.visit_count))
            best_idx = int(np.argmax(visit_counts))
            pruned = visit_counts.copy()
            for i, action in enumerate(actions):
                if i == best_idx:
                    continue
                child = root.children[int(action)]
                if child.prior <= 0.0:
                    continue
                n_forced = math.ceil(self.forced_playout_k * child.prior * sqrt_root)
                pruned[i] = max(0.0, visit_counts[i] - float(n_forced))
            visit_counts = pruned

        if temperature <= 0.0:
            max_visits = float(np.max(visit_counts))
            best_indices = np.flatnonzero(visit_counts == max_visits)
            chosen = self._sample_tied_index(best_indices)
            probs[int(actions[chosen])] = 1.0
            return probs

        adjusted = np.power(visit_counts, 1.0 / temperature)
        total = float(np.sum(adjusted))

        if total <= 0.0:
            uniform_prob = 1.0 / float(actions.size)
            probs[actions] = uniform_prob
            return probs

        dist = adjusted / total
        probs[actions] = dist
        return probs

    def get_best_move(self, board: AtaxxBoard) -> tuple[int, int, int, int] | None:
        """
        Return best move by visit count. Returns `None` when pass is best/only action.
        """
        action_probs = self.run(
            board=board,
            add_dirichlet_noise=False,
            temperature=0.0,
        )
        best_action_idx = int(np.argmax(action_probs))
        return ACTION_SPACE.decode(best_action_idx)

    def cache_stats(self) -> dict[str, float | int]:
        total = self._cache_hits + self._cache_misses
        hit_rate = (float(self._cache_hits) / float(total)) if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_entries": len(self._inference_cache),
        }
