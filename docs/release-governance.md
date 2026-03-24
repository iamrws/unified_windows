# AstraWeave Release Governance (v1)

## 1. Licensing and Attribution

Project license posture:

- AstraWeave service and SDK default license target: Apache-2.0.
- Third-party dependencies keep their original licenses.
- Every release must include machine-readable third-party attribution manifest.

## 2. Dependency and SBOM Policy

Dependency controls:

- Runtime dependencies must be version-pinned.
- Update windows are scheduled and tracked.
- Security-critical dependencies require expedited patch process.

SBOM requirements:

- Generate CycloneDX JSON SBOM per release artifact.
- SBOM must include direct and transitive dependencies.
- SBOM is release-blocking deliverable.

## 3. Signing and Integrity

Signing policy:

- All distributed binaries must be code-signed.
- Signature verification is part of release promotion.
- Unsigned artifacts cannot be promoted beyond internal test channel.
- Source-only RC bundles without compiled binaries may record signing as `not_applicable_for_source_only_rc`, but this exception does not apply once binary artifacts exist.

Integrity checks:

- Publish checksums for all release artifacts.
- Verify checksums in CI before channel promotion.

## 4. Distribution Channels

Defined channels:

- `nightly`: development-only, no production support guarantee.
- `rc`: release candidate for staged validation.
- `stable`: signed and gate-closed public release.

Promotion rules:

- Promotion requires all P0/P1 gates closed in `problems.md`.
- Stable channel additionally requires compatibility and operations gates closure.

## 5. Compliance Checklist

Required before `stable`:

- License headers and attribution report present.
- CycloneDX SBOM generated and published.
- Artifact signing verification passed.
- Release notes include API changes and known limitations.
- Security and privacy policy links included in release manifest.
