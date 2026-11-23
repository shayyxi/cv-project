# Frontend

Three Streamlit apps, one per folder, all in the FINGO colours, dark
only. The language follows the browser (French or English); `?lang=fr`
or `?lang=en` in the address forces one. Times are Eastern. There are
no theme, language or time-zone switches (reviewer decision, October
2026). `_shared.py` holds the theme, the browser-language rule and the
phone layout fix; the retraining page keeps its own theme copy and
imports the language and phone pieces.

| Folder | What it is | Start it |
|---|---|---|
| `retraining/` | Upload to Roboflow, train and compare, crop regions. Wired to `app.retraining`. `retrain_ui.py` is the menu; each page is a module in `features/` (`upload`, `training`, `crops`, `camera_settings`, plus `jobs` and `i18n`), the CSS is `theme.py`. | `retrain_ui.bat`, or `streamlit run frontend/retraining/retrain_ui.py` (port 8501) |
| `site-review/` | Site safety review: compliance per camera over a date range, the images, a heatmap, report download. Frontend only. `app.py` is the header and tabs; the pages are `features/cameras.py`, `overview.py`, `images.py`, with `state.py` (session, range, callbacks), `frames.py`, `layout.py`, `reports.py`, `i18n.py`; the CSS is `theme.py`, the logic `review_data.py`, the clickable table `table_component/`. | `site_review.bat`, or `streamlit run frontend/site-review/app.py --server.port 8502` |
| `system-health/` | AI Vision system health: cameras, processing, WordPress delivery, failures, host. Frontend only. `app.py` draws the sections in order; each section is a module in `features/` (`overview`, `performance`, `cameras`, `pipeline`, `failures`, `infrastructure`, plus `page`, `format`, `widgets`, `i18n`); the CSS is `theme.py`, the logic `health_data.py`. | `system_health.bat`, or `streamlit run frontend/system-health/app.py --server.port 8503` |

## Backend integration later

The two frontend-only apps each read one JSON file next to their
`app.py` and show the shipped sample until it exists:

- `site-review/results.json`: shape as in `sample_results.json`,
  documented at the top of `review_data.py`. Frames are looked up under
  `site-review/processed/` and the repo's `data/processed/`; a grey
  placeholder stands in for a frame that is not on the machine.
- `system-health/health.json`: shape as in `sample_health.json`,
  documented at the top of `health_data.py`. Re-read every minute and
  flagged stale when older than `stale_after_seconds`.

"Download report(s)" on the site review opens the browser's print
dialog on the current page for now; `request_report()` in
`site-review/features/reports.py` is where the report service plugs in.

The design mocks the apps replicate are in `docs/design/`. The pure
logic behind both pages is tested in `tests/test_frontend_dashboards.py`.
