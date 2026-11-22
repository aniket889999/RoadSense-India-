"""Parking API router package (sites, cameras, layouts, stability, occupancy, hazards, capacity).

Importing this package registers all routes on ``router`` in the original order and
re-exports every name the former single module defined, so existing imports keep working.
"""

from __future__ import annotations

# Everything the former module had at top level, including the names it imported
# (e.g. LayoutRevisionStatus), stays importable from this package.
from services.api.app.routers.parking._common import *  # noqa: F401,F403
from services.api.app.routers.parking._common import (  # noqa: F401
    _HAZARD_LIFECYCLE_TRANSITIONS,
    _HAZARD_REVIEW_TRANSITIONS,
    _VALID_OCCUPANCY_STATES,
    _read_bay_occupancy_summary,
    get_parking_job_manager,
    get_stability_job_manager,
    get_storage_root,
    logger,
    router,
    utc_now,
)
from services.api.app.routers.parking.sites_cameras import (  # noqa: F401
    create_camera,
    create_site,
    get_camera,
    get_reference_image,
    get_site,
    invalidate_camera_calibration,
    list_site_cameras,
    list_sites,
    upload_reference_image,
)
from services.api.app.routers.parking.layouts import (  # noqa: F401
    _build_layout_response,
    _dt_sort_key,
    create_draft_layout,
    get_layout,
    invalidate_layout,
    list_camera_layouts,
    submit_layout_for_review,
    update_layout,
    validate_layout_endpoint,
    verify_layout,
)
from services.api.app.routers.parking.stability import (  # noqa: F401
    _build_stability_response,
    acknowledge_stability_assessment,
    cancel_stability_assessment,
    create_camera_stability_assessment,
    get_camera_operational_gate,
    get_stability_assessment_detail,
    list_camera_stability_assessments,
    list_camera_stability_audit_events,
)
from services.api.app.routers.parking.occupancy_jobs import (  # noqa: F401
    _build_occupancy_job_response,
    _resolve_safe_parking_artifact,
    cancel_parking_occupancy_job,
    create_staged_upload_file,
    delete_parking_occupancy_job,
    get_parking_job_annotated_video,
    get_parking_job_manifest,
    get_parking_job_summary,
    get_parking_job_timeline,
    get_parking_occupancy_job_detail,
    list_camera_occupancy_jobs,
    submit_parking_occupancy_job,
)
from services.api.app.routers.parking.hazards_capacity import (  # noqa: F401
    create_hazard_association,
    get_camera_capacity_snapshot,
    get_hazard_association,
    list_hazard_associations,
    list_hazard_audit_events,
    list_pavement_inspections,
    record_pavement_inspection,
    review_hazard_association,
    transition_hazard_lifecycle,
)
