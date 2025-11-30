from typing import List, Optional
import numpy as np

from tracking.tracking_types import Track, TrackHistory, TrackingResult


def _build_entity_static_map(histories: List[TrackHistory]) -> dict[int, bool]:
    """
    Build mapping: entity_id -> is_static
    """
    return {h.entity_id: h.is_static for h in histories}


def _filter_histories(
    histories: List[TrackHistory],
    keep_static: bool,
) -> List[TrackHistory]:
    if keep_static:
        return [h for h in histories if h.is_static]
    else:
        return [h for h in histories if not h.is_static]


def _filter_tracks(
    tracks: List[Track],
    keep_ids: set[int],
) -> List[Track]:
    return [t for t in tracks if t.entity_id in keep_ids]


def _filter_det_to_track(
    det_to_track_per_scene: List[List[int]],
    entity_is_static: dict[int, bool],
    keep_static: bool,
) -> List[List[int]]:
    """
    Filter detection-to-track mappings so that only desired entities remain.
    Unwanted entity IDs become -1.
    """
    filtered: List[List[int]] = []

    for det_list in det_to_track_per_scene:
        new_list: List[int] = []
        for ent_id in det_list:
            if ent_id == -1:
                new_list.append(-1)
                continue

            is_static = entity_is_static.get(ent_id, False)

            if keep_static:
                new_list.append(ent_id if is_static else -1)
            else:
                new_list.append(ent_id if not is_static else -1)

        filtered.append(new_list)

    return filtered


def _filter_point_to_entity(
    point_to_entity_per_scene: Optional[List[np.ndarray]],
    entity_is_static: dict[int, bool],
    keep_static: bool,
) -> Optional[List[np.ndarray]]:
    if point_to_entity_per_scene is None:
        return None

    filtered: List[np.ndarray] = []
    for mapping in point_to_entity_per_scene:
        m = mapping.copy()
        for i, ent_id in enumerate(m):
            if ent_id == -1:
                continue

            is_static = entity_is_static.get(int(ent_id), False)

            if keep_static and not is_static:
                m[i] = -1
            if not keep_static and is_static:
                m[i] = -1

        filtered.append(m)

    return filtered


def filter_tracking_result(
    result: TrackingResult,
    keep_static: bool,
) -> TrackingResult:
    """
    Return a new TrackingResult containing only STATIC or only DYNAMIC entities.

    keep_static = True  → keep only static entities
    keep_static = False → keep only dynamic entities
    """
    entity_is_static = _build_entity_static_map(result.histories)

    # histories
    filtered_histories = _filter_histories(result.histories, keep_static)
    keep_ids = {h.entity_id for h in filtered_histories}
    filtered_tracks = _filter_tracks(result.tracks, keep_ids)
    det_filtered = _filter_det_to_track(
        result.det_to_track_per_scene,
        entity_is_static,
        keep_static=keep_static,
    )

    pte_filtered = _filter_point_to_entity(
        result.point_to_entity_per_scene,
        entity_is_static,
        keep_static=keep_static,
    )

    return TrackingResult(
        det_to_track_per_scene=det_filtered,
        point_to_entity_per_scene=pte_filtered,
        tracks=filtered_tracks,
        histories=filtered_histories,
    )
