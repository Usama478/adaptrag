# 0003: File hashing and storage

## Chosen
SHA-256 (Python's built-in hashlib) as the file fingerprint. Uploaded files are
saved as data/uploads/<hash>.pdf, and only after all validation checks pass.

## Rejected
- MD5 / SHA-1: broken; two different files can be made to share a hash.
- Python's hash(): changes every time Python restarts, so it can't be stored.
- Saving by the user's file name: names collide, and a name like ../../x.pdf
  could write outside the upload folder.

## Why
Same content always gives the same name, so the same file is stored once.
The hash is also the key for duplicate detection (Step 4) and the embedding
cache (Stage 2).

## Known limit
In Docker, saved files are lost when the container is removed. Add a volume
before deployment.