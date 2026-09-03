from pydantic import ValidationError

from app.schemas.product import ImportError


def validation_reason(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )


def import_error(row_number: int, code: str, reason: str) -> ImportError:
    return ImportError(row_number=row_number, code=code, reason=reason)
