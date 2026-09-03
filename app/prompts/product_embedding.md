# Product Embedding Content Template v2

`embedding_service.build_embedding_content(product)` creates one product-level
retrieval chunk (`chunk_index = 0`) and includes every sellable variant.

```text
Tên sản phẩm: {name}. Mô tả: {description}. Danh mục: {category}.
Tags: {tags}. Trạng thái sản phẩm: {status}.
Biến thể: Biến thể 1; Tên: {variant_name}; SKU: {sku};
Thuộc tính: {attributes}; Giá: {price} {currency};
Giá gốc: {original_price} {currency}; Tồn kho: {stock_quantity};
Trạng thái kho: {stock_status} | ...
Metadata: {product_metadata}.
```

Rules:

- Product-level fields come from `products`; price, stock and options come from
  `product_variants`.
- Render all variants in stable input/database order.
- Do not emit Python/JSON null sentinels (`None`, `null`, `undefined`).
- Keep the final text within the 500-token MVP budget.
- Hash the final UTF-8 content with SHA-256 and store it as `content_hash`.
- Store `format_version = product-v1`, `normalization_version = nfc-v1`
  and the configured vector dimension in embedding metadata.
