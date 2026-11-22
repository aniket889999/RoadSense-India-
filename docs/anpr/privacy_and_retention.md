# RoadSense ANPR privacy and retention card

## Scope and observed behavior

This card covers the still-image ANPR path and its explicit review exports. It
does not certify the complete RoadSense platform or prescribe a legal retention
period. The [demo guide](demo_and_jury_guide.md) and
[image upload contract](development_plan.md#module-2-file-ownership-and-upload-contract)
describe the current non-persisting endpoint and review workflow.

| Data boundary | Current behavior | Retention implication |
| --- | --- | --- |
| Recognition request | Raw JPEG/PNG bytes are decoded and processed locally. The ANPR endpoint does not write uploaded images, crops or OCR text to files or its database. | Input/result buffers exist in process memory; this is not a secure-memory-erasure guarantee. |
| Browser inspection | The selected image, preview and returned readings are held in component state. Preview object URLs are revoked on replacement/unmount. | Browser/OS caches, swap, screenshots and backups are outside this application-level guarantee. |
| Explicit annotated PNG download | The browser exports the source scene with boxes and candidate text, marked unverified. | The downloaded image can reveal readable plates and bystanders; it is not an anonymized export. |
| Explicit review JSON download | The browser exports full raw/normalized reading results, review cues and request metadata with `human_verified: false`. | The user's saved file persists independently of the API. Downloading or displaying it does not mark it human verified. |
| Research artifacts | Local source/prepared datasets, OCR predictions, training outputs and optional per-plate failure details persist separately from the request endpoint. | Git exclusion does not expire, encrypt or delete these files. Aggregate reports must not be assumed to sanitize detailed companion files. |

Evidence: [demo exports](demo_and_jury_guide.md#reproduce-the-local-demo),
[upload contract](development_plan.md#module-2-file-ownership-and-upload-contract),
[dataset intake](witvix_intake_results.md), and
[failure-analysis aggregate/details distinction](ocr_failure_analysis.md).
The behavior was also checked against the current
[API route](../../services/api/app/routers/anpr.py),
[browser inspection component](../../apps/web/src/components/anpr/AnprInspectionPanel.tsx)
and [export helpers](../../apps/web/src/lib/anprExports.ts).

## Local execution boundary

The provided [local launcher](../../scripts/serve_anpr_local.py) binds the API to
loopback and composes existing local detector/OCR assets. The selected RapidOCR
recognizer uses CPU ONNX execution; the recorded detector uses explicit MPS.
Missing assets do not authorize downloads or cloud OCR. Model/package
provisioning is a separate activity from handling an image request. See the
[current launch guide](demo_and_jury_guide.md) and [model card](model_card.md).

This is a claim about the inspected ANPR path, not proof of zero network traffic
from the browser, dependencies, IDE, host or other platform modules. Loopback and
CORS restrictions are not user authentication. The ANPR route currently has no
route-specific API-key/rate-limit control or automatic export-retention service;
these controls must not be advertised as delivered by this documentation.

The route's adapter-error log records a request identifier rather than image
bytes or plate strings. Deployment access logs, proxies and debugging tools need
their own review. Do not enable request-body or OCR-text logging. Research
details/prediction files can contain registration strings even when an aggregate
report does not; keep them outside Git and public evidence bundles.

## Personal-data consideration

Plate strings and vehicle imagery may relate to an identifiable individual;
treat them as potentially personal data when assessing an Indian deployment.
The [Digital Personal Data Protection Act, 2023, sections 2(t) and 2(n)](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf)
defines personal data by identifiability and digital personal data by its digital
form. Applying that definition to a particular capture/lookup context requires
assessment; not every vehicle image automatically identifies a person. Check
applicable commencement notifications and current rules before deployment.
This is a design consideration, not legal advice or a compliance determination.

## Recommendations for a deployment — not implemented controls

- Document the collection purpose and lawful basis, provide appropriate notice,
  obtain consent where required, and check permissions for both captured images
  and research data. Dataset licensing and privacy obligations are separate;
  see the [data card](data_card.md).
- Restrict operators and export access, keep secrets outside Git, protect
  transport/storage when serving beyond loopback, and audit access using minimal
  metadata rather than full readings. A restricted local account is not a
  substitute for deployment authentication and authorization.
- Assign an owner to each stored class: source images, prepared copies,
  prediction/detail files, exported PNG/JSON, logs and backups. Define a justified
  retention period and deletion/exception procedure for each, including derived
  copies. No automatic ANPR purge period is currently enforced or claimed.
- Prefer permitted/de-identified demonstration material. Review images and text
  before sharing; the unverified label on an export does not redact identifiers.
- Keep uncertain readings in human review. Registration lookup, identity-to-bay
  association, payment and barrier actuation remain outside the implemented ANPR
  workflow; do not repurpose candidate text for those decisions without a
  separately validated and authorized system.

These are recommendations for future operational work. This task does not
change storage behavior, add authentication, delete artifacts or inspect the
held-out ANPR images.
