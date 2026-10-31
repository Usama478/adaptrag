"""Print every document: id, filename, status, page count, and whether its file exists.

The file check looks in data_dir / "uploads" of THIS process (as in routes/documents.py).
The local app and the Docker app share the database but not the upload folder, so a
document uploaded through Docker shows "file MISSING" here. `scripts/reingest.py
DOCUMENT_ID --file PATH` puts the file back.
Reads the database URL from settings (ADAPTRAG_DATABASE_URL or .env).

Usage: python scripts/list_documents.py
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from adaptrag.config import get_settings
from adaptrag.db import get_engine
from adaptrag.models import Document
from adaptrag.storage import upload_path


def main() -> None:
    settings = get_settings()
    engine = get_engine(settings.database_url)
    upload_dir = settings.data_dir / "uploads"

    print(f"files checked in {upload_dir}")
    with Session(engine) as session:
        for document in session.scalars(select(Document).order_by(Document.id)):
            has_file = upload_path(document.file_hash, upload_dir).exists()
            print(
                f"document {document.id}: {document.filename}, status {document.status}, "
                f"pages {document.page_count}, file {'ok' if has_file else 'MISSING'}"
            )


if __name__ == "__main__":
    main()
