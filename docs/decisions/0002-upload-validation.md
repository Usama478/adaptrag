# 0002: Upload validation

## Chosen
Check the file extension (.pdf), the size (20 MB, set by env var), and the first
bytes of the file (`%PDF-`). Return 415 for the wrong type and 413 for too large.

## Rejected
- Trusting only the file name: anyone can rename any file to .pdf.
- Trusting the Content-Type header: the client sets it, so it can lie.
- No size limit: one huge upload could exhaust memory.

## Why
Cheap checks at the front door stop bad files before any slow work happens.

## Known limit
The whole file is read into memory. Fine at 20 MB; revisit if the limit grows.