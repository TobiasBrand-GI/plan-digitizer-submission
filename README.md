# PlanDigitalizer
Automatically digitalize features from German zoning plans

## Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with the Compose plugin (`docker compose`)
- A [Mistral AI](https://console.mistral.ai/) API key (used for text classification)
- The ALKIS land-use shapefile `nutzungFlurstueck_kgt.shp` (with its `.shx`, `.dbf`, `.prj` files)

> [!IMPORTANT]
> Example ALKIS files and other geodata are available under: [https://1drv.ms/f/c/c7dff35ae91c49d4/IgAP84ogdOcrSpqd-Gtv1hNsAYgKMsRr1KE3bky_pJ1MH84?e=402uU5](https://1drv.ms/f/c/c7dff35ae91c49d4/IgAP84ogdOcrSpqd-Gtv1hNsAYgKMsRr1KE3bky_pJ1MH84?e=402uU5)

### Setup

1. Clone the repository and change into it:
   ```bash
   git clone https://github.com/TobiasBrand-GI/plan-digitizer-submission.git
   cd plan-digitizer-submission
   ```
2. Copy the example environment file and adjust it:
   ```bash
   cp .env.example .env
   ```
   In `.env`, replace `<KEY>` with your Mistral API key. `MISTRAL_REQUEST_DELAY_MS` (pause between API requests) and `MISTRAL_MAX_RETRIES` (retries of a failed request) can be left at their defaults.
3. If needed, replace the ALKIS shapefile in `vectors/`:
   ```
   vectors/nutzungFlurstueck_kgt.shp   (+ .shx, .dbf, .prj)
   ```
4. Build and start the stack:
   ```bash
   docker compose up -d --build
   ```

### Usage

- **Web UI:** open <http://localhost:8000/>, drop the zoning plan (GeoTIFF) and its boundary (Umring, GeoJSON), enter the AGS and municipality name, and start processing. When the run is finished, the complete result folder (including `plan.gml` and `export_report.json`) can be downloaded as a ZIP.
- **API documentation:** <http://localhost:8000/docs>
- The system itself states that it listens on http://0.0.0.0:8000, which can lead to issues, if this IP is not mapped as *localhost* on the local machine.

## License

- **Code:** GNU Affero General Public License v3.0 (see [`LICENSE`](LICENSE)).
- **Research data** in [`data/test_results`](data/test_results): Creative Commons Attribution 4.0 International (CC BY 4.0, see [`data/LICENSE`](data/LICENSE)).
- **Third-party material keeps its own license:** XPlanung schemas in `schemas/` (XLeitstelle), ALKIS data (dl-de/zero-2-0), zoning plans (respective municipalities).
  - Due to file size, plans, ALKIS and other geodata files are available here: [https://1drv.ms/f/c/c7dff35ae91c49d4/IgAP84ogdOcrSpqd-Gtv1hNsAYgKMsRr1KE3bky_pJ1MH84?e=402uU5](https://1drv.ms/f/c/c7dff35ae91c49d4/IgAP84ogdOcrSpqd-Gtv1hNsAYgKMsRr1KE3bky_pJ1MH84?e=402uU5)
