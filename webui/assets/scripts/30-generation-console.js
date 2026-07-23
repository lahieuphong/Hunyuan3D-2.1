
        let generationConsoleTimer = null;
        let generationConsoleUid = null;
        let generationConsoleStartedAt = null;
        let generationConsoleSeenEvents = new Set();
        let generationConsoleParamsRendered = false;
        let generationConsolePollMisses = 0;

        const generationConsoleStageLevels = {
            request_received: "QUEUE",
            validating_input: "CHECK",
            input_validated: "INPUT",
            input_saved: "STORE",
            preprocessing_input: "PREP",
            input_ready: "READY",
            shape_generation: "CUDA",
            prepare_conditioning: "IMAGE",
            encode_conditioning: "ENCODE",
            conditioning_ready: "COND",
            prepare_timestep_schedule: "SCHED",
            latents_initialized: "LATENT",
            diffusion_started: "CUDA",
            diffusion_step: "STEP",
            diffusion_completed: "CUDA",
            vae_decoding: "VAE",
            volume_decoding: "VOLUME",
            volume_decoding_progress: "VOLUME",
            volume_decoding_completed: "VOLUME",
            surface_extraction: "OCTREE",
            surface_extraction_completed: "MESH",
            trimesh_conversion: "MESH",
            extracting_mesh: "MESH",
            mesh_ready: "MESH",
            exporting_glb: "WRITE",
            building_preview: "VIEW",
            completed: "DONE",
            failed: "ERROR",
        };

        const generationConsoleElement = (id) => document.getElementById(id);

        const generationConsoleElapsed = (timestamp = null) => {
            if (!generationConsoleStartedAt) return "+00.0s";
            const target = timestamp ? new Date(timestamp).getTime() : Date.now();
            const elapsed = Math.max(0, (target - generationConsoleStartedAt) / 1000);
            return "+" + elapsed.toFixed(1).padStart(4, "0") + "s";
        };

        const appendGenerationConsoleLine = (level, message, kind = "info", timestamp = null) => {
            const log = generationConsoleElement("generation-console-log");
            if (!log) return;

            const line = document.createElement("div");
            line.className = "generation-console-line";
            line.dataset.kind = kind;

            const time = document.createElement("span");
            time.className = "generation-console-time";
            time.textContent = generationConsoleElapsed(timestamp);

            const levelElement = document.createElement("span");
            levelElement.className = "generation-console-level";
            levelElement.textContent = level;

            const messageElement = document.createElement("span");
            messageElement.className = "generation-console-message";
            messageElement.textContent = message;

            line.append(time, levelElement, messageElement);
            log.appendChild(line);
            while (log.children.length > 200) log.firstElementChild?.remove();
            log.scrollTop = log.scrollHeight;
        };

        const setGenerationConsoleProgress = (progress, stage) => {
            const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
            const bar = generationConsoleElement("generation-console-progress");
            const percent = generationConsoleElement("generation-console-percent");
            const stageElement = generationConsoleElement("generation-console-stage");
            if (bar) bar.style.width = safeProgress + "%";
            if (percent) percent.textContent = Math.round(safeProgress) + "%";
            if (stageElement && stage) stageElement.textContent = stage;
        };

        const setGenerationConsoleState = (state, label) => {
            const root = generationConsoleElement("generation-console");
            const status = generationConsoleElement("generation-console-status");
            if (root) root.dataset.state = state;
            if (status) status.textContent = label;
        };

        const setGenerationDetail = (id, value) => {
            const element = generationConsoleElement(id);
            if (element) element.textContent = value ?? "—";
        };

        const formatGenerationCount = (value) => {
            const number = Number(value);
            if (!Number.isFinite(number)) return "—";
            if (number >= 1000000) return (number / 1000000).toFixed(2).replace(/\.00$/, "") + "M";
            if (number >= 1000) return (number / 1000).toFixed(1).replace(/\.0$/, "") + "K";
            return Math.round(number).toLocaleString("en-US");
        };

        const formatGenerationBytes = (value) => {
            const bytes = Number(value);
            if (!Number.isFinite(bytes) || bytes <= 0) return null;
            if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
            if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
            return bytes + " B";
        };

        const resetGenerationDetails = (mode = "") => {
            setGenerationDetail("generation-info-model", "—");
            setGenerationDetail("generation-info-views", mode === "4-VIEW" ? "4" : mode === "1-VIEW" ? "1" : "—");
            setGenerationDetail("generation-info-time", "—");
            setGenerationDetail("generation-info-resolution", "—");
            setGenerationDetail("generation-info-polygons", "—");
            setGenerationDetail("generation-info-vertices", "—");
            setGenerationDetail("generation-output-meta", "Awaiting generated mesh");
        };

        const updateGenerationDetails = (manifest) => {
            const params = manifest.params || manifest.stats?.params || {};
            const stats = manifest.stats || {};
            const rawModel = manifest.model?.shapegen || stats.model?.shapegen || "";
            const modelName = String(rawModel).split("/").filter(Boolean).at(-1) || "—";
            const viewCount = Array.isArray(params.views_used)
                ? params.views_used.length
                : params.input_mode === "four" ? 4 : params.input_mode ? 1 : "—";
            const totalSeconds = Number(stats.time?.total);
            const faces = stats.number_of_faces ?? manifest.number_of_faces;
            const vertices = stats.number_of_vertices ?? manifest.number_of_vertices;

            setGenerationDetail("generation-info-model", modelName);
            setGenerationDetail("generation-info-views", String(viewCount));
            setGenerationDetail(
                "generation-info-time",
                Number.isFinite(totalSeconds) ? totalSeconds.toFixed(1) + " s" : "—"
            );
            setGenerationDetail("generation-info-resolution", params.octree_resolution ?? "—");
            setGenerationDetail("generation-info-polygons", formatGenerationCount(faces));
            setGenerationDetail("generation-info-vertices", formatGenerationCount(vertices));

            const outputMeta = generationConsoleElement("generation-output-meta");
            if (!outputMeta) return;
            if (manifest.status !== "completed") {
                outputMeta.textContent = "Generation in progress";
                return;
            }

            const uid = manifest.generation_uid;
            const meshFilename = String(manifest.outputs?.mesh || "white_mesh.glb");
            outputMeta.textContent = "GLB · saved to source";
            fetch(
                "/static/" + encodeURIComponent(uid) + "/" + encodeURIComponent(meshFilename),
                {method: "HEAD", cache: "no-store"}
            ).then((response) => {
                if (!response.ok || generationConsoleUid !== uid) return;
                const size = formatGenerationBytes(response.headers.get("content-length"));
                if (size) outputMeta.textContent = "GLB · " + size + " · saved";
            }).catch(() => {});
        };

        const stopGenerationConsolePolling = () => {
            if (generationConsoleTimer !== null) {
                window.clearInterval(generationConsoleTimer);
                generationConsoleTimer = null;
            }
        };

        const renderGenerationManifest = (manifest) => {
            if (!manifest || manifest.generation_uid !== generationConsoleUid) return;

            updateGenerationDetails(manifest);

            if (
                manifest.storage_folder
                && !generationConsoleSeenEvents.has("__storage__")
            ) {
                generationConsoleSeenEvents.add("__storage__");
                appendGenerationConsoleLine(
                    "STORE",
                    "Target: " + manifest.storage_folder,
                    "muted"
                );
            }

            if (manifest.params && !generationConsoleParamsRendered) {
                generationConsoleParamsRendered = true;
                const params = manifest.params;
                appendGenerationConsoleLine(
                    "CONFIG",
                    "steps=" + params.steps
                        + " guidance=" + params.guidance_scale
                        + " octree=" + params.octree_resolution
                        + " chunks=" + params.num_chunks
                        + " seed=" + params.seed,
                    "command"
                );
            }

            (manifest.events || []).forEach((event) => {
                const eventKey = event.stage + "|" + event.at;
                if (generationConsoleSeenEvents.has(eventKey)) return;
                generationConsoleSeenEvents.add(eventKey);
                const kind = event.stage === "completed"
                    ? "success"
                    : event.stage === "failed" ? "error" : "info";
                appendGenerationConsoleLine(
                    generationConsoleStageLevels[event.stage] || "INFO",
                    event.message || event.stage,
                    kind,
                    event.at
                );
            });

            setGenerationConsoleProgress(
                manifest.progress,
                (manifest.events || []).at(-1)?.message || manifest.stage
            );

            const clock = generationConsoleElement("generation-console-clock");
            if (clock) clock.textContent = "LIVE " + generationConsoleElapsed();

            if (manifest.status === "completed") {
                const stats = manifest.stats || {};
                if (!generationConsoleSeenEvents.has("__mesh_stats__")) {
                    generationConsoleSeenEvents.add("__mesh_stats__");
                    appendGenerationConsoleLine(
                        "STATS",
                        "vertices=" + (stats.number_of_vertices ?? "-")
                            + " faces=" + (stats.number_of_faces ?? "-")
                            + " total=" + Number(stats.time?.total || 0).toFixed(2) + "s",
                        "success"
                    );
                    const storageFolder = String(
                        manifest.storage_folder || ("/static/" + generationConsoleUid)
                    ).replace(/[\/]+$/, "");
                    const meshFilename = String(manifest.outputs?.mesh || "white_mesh.glb");
                    appendGenerationConsoleLine(
                        "OUTPUT",
                        storageFolder + "/" + meshFilename,
                        "success"
                    );
                }
                setGenerationConsoleState("completed", "COMPLETED");
                setGenerationConsoleProgress(100, "3D model is ready");
                if (clock) clock.textContent = "SAVED TO SOURCE";
                stopGenerationConsolePolling();
            } else if (manifest.status === "failed") {
                if (!generationConsoleSeenEvents.has("__error__")) {
                    generationConsoleSeenEvents.add("__error__");
                    appendGenerationConsoleLine(
                        "ERROR",
                        String(manifest.error || "Unknown generation error").replace(/^'|'$/g, ""),
                        "error"
                    );
                }
                setGenerationConsoleState("failed", "FAILED");
                setGenerationConsoleProgress(100, "Generation stopped with an error");
                if (clock) clock.textContent = "ERROR SAVED TO MANIFEST";
                stopGenerationConsolePolling();
            } else {
                setGenerationConsoleState("running", "RUNNING");
            }
        };

        const pollGenerationManifest = async () => {
            const uid = generationConsoleUid;
            if (!uid) return;
            try {
                const response = await fetch(
                    "/static/" + encodeURIComponent(uid) + "/generation.json?t=" + Date.now(),
                    {cache: "no-store"}
                );
                if (!response.ok) {
                    generationConsolePollMisses += 1;
                    if (generationConsolePollMisses === 4) {
                        appendGenerationConsoleLine("QUEUE", "Waiting for the backend worker...", "muted");
                        setGenerationConsoleProgress(2, "Waiting in the Gradio queue");
                    }
                    return;
                }
                generationConsolePollMisses = 0;
                renderGenerationManifest(await response.json());
            } catch (error) {
                generationConsolePollMisses += 1;
                if (generationConsolePollMisses === 8) {
                    appendGenerationConsoleLine("WARN", "Manifest polling will retry automatically", "muted");
                }
            }
        };

        const startGenerationConsole = (uid, resumed = false) => {
            const root = generationConsoleElement("generation-console");
            if (!root || !uid) return;

            stopGenerationConsolePolling();
            generationConsoleUid = uid;
            generationConsoleStartedAt = Date.now();
            generationConsoleSeenEvents = new Set();
            generationConsoleParamsRendered = false;
            generationConsolePollMisses = 0;

            const log = generationConsoleElement("generation-console-log");
            if (log) log.replaceChildren();
            const job = generationConsoleElement("generation-console-job");
            if (job) job.textContent = "generation/" + uid;
            const mode = currentAppUrl().searchParams.get("tab") === "multi-view" ? "4-VIEW" : "1-VIEW";
            const modeElement = generationConsoleElement("generation-console-mode");
            if (modeElement) modeElement.textContent = mode;
            const clock = generationConsoleElement("generation-console-clock");
            if (clock) clock.textContent = "CONNECTING TO MANIFEST";
            resetGenerationDetails(mode);

            setGenerationConsoleState("running", resumed ? "RESTORING" : "STARTING");
            setGenerationConsoleProgress(1, resumed ? "Restoring generation state" : "Dispatching request");
            appendGenerationConsoleLine(
                resumed ? "RESUME" : "$",
                (resumed ? "restore" : "hunyuan3d.generate")
                    + " --mode " + mode.toLowerCase()
                    + " --uid " + uid,
                "command"
            );
            appendGenerationConsoleLine(
                "STORE",
                "Target: waiting for generation manifest",
                "muted"
            );

            window.setTimeout(pollGenerationManifest, 120);
            generationConsoleTimer = window.setInterval(pollGenerationManifest, 700);
        };

        const syncGenerationConsoleFromUrl = () => {
            const uid = currentAppUrl().searchParams.get("generation");
            if (uid && uid !== generationConsoleUid) startGenerationConsole(uid, true);
        };

        const beginGeneration = () => {
            const url = currentAppUrl();
            const uid = createGenerationUid();
            url.searchParams.set("generation", uid);
            window.history.pushState({}, "", url);
            activeGenerationRouteUid = uid;
            startGenerationConsole(uid);
        };

        const installGenerationRouting = () => {
            const buttonRoot = document.getElementById("generate-3d-button");
            if (!buttonRoot || buttonRoot.dataset.generationRouteWired === "true") return;
            buttonRoot.dataset.generationRouteWired = "true";
            buttonRoot.addEventListener("click", beginGeneration, {capture: true});
        };
