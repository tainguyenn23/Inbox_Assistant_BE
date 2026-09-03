from typing import Any

from pydantic import ValidationError

from app.core.config import settings
from app.importers._common import import_error, validation_reason
from app.schemas.import_product import NormalizedProduct
from app.schemas.product import ImportError


def parse_json_import(
    data: list[Any],
) -> tuple[list[NormalizedProduct], list[ImportError]]:
    if len(data) > settings.max_import_rows:
        return [], [
            import_error(
                settings.max_import_rows + 1,
                "MAX_IMPORT_ROWS_EXCEEDED",
                f"Import supports at most {settings.max_import_rows} rows",
            )
        ]

    products: list[NormalizedProduct] = []
    errors: list[ImportError] = []
    for row_number, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            errors.append(
                import_error(
                    row_number,
                    "INVALID_ROW",
                    "Row must be a JSON object",
                )
            )
            continue
        try:
            products.append(NormalizedProduct.model_validate(row))
        except ValidationError as exc:
            errors.append(
                import_error(
                    row_number,
                    "VALIDATION_ERROR",
                    validation_reason(exc),
                )
            )
    return products, errors
