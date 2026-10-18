The two duplicate review ZIPs were removed during user-authorized cleanup.
Their original external manifests remain here unchanged. All member bytes,
including historical versions, are retained in the project and mapped by
../archives/ORGANIZATION_MANIFEST.json.

From the project root:
  python scripts/organized_review.py --restore revision --output /tmp/otg_revision_review.zip
  python scripts/organized_review.py --restore original --output /tmp/otg_original_review.zip

Restored members have the original hashes. ZIP metadata/compression and the
archive checksum can differ. Existing historical verification reports remain
records of the original ZIPs. Do not overwrite them merely to record cleanup.
