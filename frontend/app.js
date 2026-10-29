// Two views, switched by the URL hash: "#/" (form) and "#/job/<id>"
// (progress). The progress view polls GET /jobs/<id> until the job is
// done or failed.
const POLL_MS = 2000;

const jobIdFromHash = () => {
  const match = location.hash.match(/^#\/job\/(.+)$/);
  return match ? decodeURIComponent(match[1]) : null;
};

Vue.createApp({
  data() {
    return {
      jobId: jobIdFromHash(),
      slots: [
        { key: "tif", label: "Zoning plan (GeoTIFF)", suffixes: [".tif", ".tiff"], over: false },
        { key: "umring", label: "Plan boundary (Umring)", suffixes: [".geojson", ".json"], over: false },
      ],
      files: { tif: null, umring: null },
      ags: "",
      gemeindeName: "",
      uploading: false,
      status: null,
      error: null,
      timer: null,
    };
  },

  computed: {
    progress() {
      if (!this.status) return 0;
      if (this.status.state === "done") return 100;
      // steps before the current one count as finished
      return Math.max(0, this.status.step - 1) / this.status.total_steps * 100;
    },
    currentStepName() {
      if (!this.status || !this.status.step) return "Starting…";
      return `Step ${this.status.step} of ${this.status.total_steps}: ${this.status.steps[this.status.step - 1]}…`;
    },
  },

  methods: {
    setFile(slot, file) {
      slot.over = false;
      if (!file) return;
      if (!slot.suffixes.some((s) => file.name.toLowerCase().endsWith(s))) {
        this.error = `${file.name}: expected ${slot.suffixes.join(" or ")}`;
        return;
      }
      this.error = null;
      this.files[slot.key] = file;
    },
    onDrop(slot, event) {
      this.setFile(slot, event.dataTransfer.files[0]);
    },
    formatSize(bytes) {
      return bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.ceil(bytes / 1e3)} KB`;
    },

    async submit() {
      this.uploading = true;
      this.error = null;
      const body = new FormData();
      body.append("tif", this.files.tif);
      body.append("umring", this.files.umring);
      body.append("ags", this.ags);
      body.append("gemeinde_name", this.gemeindeName);
      try {
        const response = await fetch("/jobs", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || response.statusText);
        location.hash = `#/job/${encodeURIComponent(data.job_id)}`;
      } catch (e) {
        this.error = `Could not start processing: ${e.message}`;
      } finally {
        this.uploading = false;
      }
    },

    async poll() {
      try {
        const response = await fetch(`/jobs/${encodeURIComponent(this.jobId)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || response.statusText);
        this.status = data;
        this.error = null;
        if (data.state !== "running") return; // finished -- stop polling
      } catch (e) {
        this.error = `Could not fetch status: ${e.message}`;
      }
      this.timer = setTimeout(this.poll, POLL_MS);
    },

    route() {
      clearTimeout(this.timer);
      this.jobId = jobIdFromHash();
      this.status = null;
      this.error = null;
      if (this.jobId) this.poll();
    },
  },

  mounted() {
    window.addEventListener("hashchange", this.route);
    this.route();
  },
}).mount("#app");
