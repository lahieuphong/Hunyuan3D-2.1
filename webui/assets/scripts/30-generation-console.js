
        let generationConsoleTimer = null;
        let generationConsoleUid = null;
        let generationConsoleStartedAt = null;
        let generationConsoleSeenEvents = new Set();
        let generationConsoleParamsRendered = false;
        let generationConsolePollMisses = 0;
        let generationConsoleLastManifest = null;
        let generationConsoleMode = "1-VIEW";
        let generationConsoleResumed = false;

        const generationConsoleStageLevels = {
            request_received: ["generation.console.level.queue", "QUEUE"],
            validating_input: ["generation.console.level.check", "CHECK"],
            input_validated: ["generation.console.level.input", "INPUT"],
            input_saved: ["generation.console.level.store", "STORE"],
            preprocessing_input: ["generation.console.level.prep", "PREP"],
            input_ready: ["generation.console.level.ready", "READY"],
            shape_generation: ["generation.console.level.cuda", "CUDA"],
            prepare_conditioning: ["generation.console.level.image", "IMAGE"],
            encode_conditioning: ["generation.console.level.encode", "ENCODE"],
            conditioning_ready: ["generation.console.level.conditioning", "COND"],
            prepare_timestep_schedule: ["generation.console.level.schedule", "SCHED"],
            latents_initialized: ["generation.console.level.latent", "LATENT"],
            diffusion_started: ["generation.console.level.cuda", "CUDA"],
            diffusion_step: ["generation.console.level.step", "STEP"],
            diffusion_completed: ["generation.console.level.cuda", "CUDA"],
            vae_decoding: ["generation.console.level.vae", "VAE"],
            volume_decoding: ["generation.console.level.volume", "VOLUME"],
            volume_decoding_progress: ["generation.console.level.volume", "VOLUME"],
            volume_decoding_completed: ["generation.console.level.volume", "VOLUME"],
            surface_extraction: ["generation.console.level.octree", "OCTREE"],
            surface_extraction_completed: ["generation.console.level.mesh", "MESH"],
            trimesh_conversion: ["generation.console.level.mesh", "MESH"],
            extracting_mesh: ["generation.console.level.mesh", "MESH"],
            mesh_ready: ["generation.console.level.mesh", "MESH"],
            exporting_glb: ["generation.console.level.write", "WRITE"],
            preparing_texture_inputs: ["generation.console.level.color", "COLOR"],
            baking_original: ["generation.console.level.bake", "BAKE"],
            coloring_original_fallback: ["generation.console.level.color", "COLOR"],
            scoring_original: ["generation.console.level.score", "SCORE"],
            original_ready: ["generation.console.level.color", "COLOR"],
            building_preview: ["generation.console.level.view", "VIEW"],
            completed: ["generation.console.level.done", "DONE"],
            failed: ["generation.console.level.error", "ERROR"],
            face_hair_rc_selected: ["generation.console.level.score", "SCORE"],
        };

        const generationConsoleFormat = (template, params = {}) => String(template).replace(
            /\{([A-Za-z0-9_]+)\}/g,
            (match, name) => Object.prototype.hasOwnProperty.call(params, name)
                ? String(params[name])
                : match
        );

        const generationConsoleT = (key, params = {}, fallback = key) => {
            if (typeof uiT === "function") {
                const translated = uiT(key, params);
                if (translated !== key) return translated;
            }
            return generationConsoleFormat(fallback, params);
        };

        const generationConsoleLocale = () => (
            typeof currentUiLocale === "function" ? currentUiLocale() : "en"
        );

        const generationConsoleModeLabel = (mode) => {
            const key = {
                "1-VIEW": "generation.console.mode.single",
                "4-VIEW": "generation.console.mode.four",
                "6-VIEW": "generation.console.mode.six",
                "10-VIEW": "generation.console.mode.ten",
            }[mode];
            return key ? generationConsoleT(key, {}, mode) : mode;
        };

        const generationConsoleValue = (value, fallback = "-") => {
            if (value === undefined || value === null || value === "") return fallback;
            if (Array.isArray(value) || (typeof value === "object" && value !== null)) {
                try {
                    return JSON.stringify(value);
                } catch {
                    return String(value);
                }
            }
            return String(value);
        };

        const generationConsoleLevel = (stage, fallback = "INFO") => {
            const definition = generationConsoleStageLevels[stage];
            if (!definition) {
                return generationConsoleT("generation.console.level.info", {}, fallback);
            }
            return generationConsoleT(definition[0], {}, definition[1]);
        };

        const generationConsoleStageFallbacks = {
            request_received: "Generation request accepted",
            validating_input: "Validating input payload",
            input_validated: "Input validation completed",
            input_saved: "Input snapshots saved to source storage",
            preprocessing_input: "Preprocessing input views",
            input_ready: "Input tensor is ready for inference",
            shape_generation: "Starting Hunyuan3D inference pipeline",
            prepare_conditioning: "Preparing image conditioning tensors",
            encode_conditioning: "Encoding vision conditioning features",
            conditioning_ready: "Vision conditioning is ready",
            prepare_timestep_schedule: "Building diffusion timestep schedule",
            latents_initialized: "Latent noise tensor initialized",
            diffusion_started: "Diffusion sampling started",
            diffusion_completed: "Diffusion sampling completed",
            vae_decoding: "Decoding latent representation with ShapeVAE",
            volume_decoding: "Starting dense volume decoding",
            volume_decoding_progress: "Decoding dense volume chunks",
            volume_decoding_completed: "Dense volume decoding completed",
            surface_extraction: "Running marching-cubes surface extraction",
            surface_extraction_completed: "Surface extraction completed",
            trimesh_conversion: "Converting generated surface to Trimesh",
            extracting_mesh: "Converting generated surface to Trimesh",
            mesh_ready: "Mesh geometry is ready",
            exporting_glb: "Exporting white GLB",
            preparing_texture_inputs: "Preparing images for color projection",
            baking_original: "Baking the colored Original GLB",
            coloring_original_fallback: "Building the colored Original fallback",
            scoring_original: "Scoring face and hair consistency before publication",
            original_ready: "Colored Original GLB is ready",
            building_preview: "Building interactive 3D preview",
            completed: "Generation completed successfully",
            failed: "Generation failed",
            face_hair_rc_selected: (
                "Selected the clearest fixed-geometry face/hair texture "
                + "candidate after ten-view RC"
            ),
        };

        const generationConsoleEventDetail = (event, name) => (
            event?.[name]
            ?? event?.pipeline_stage?.[name]
            ?? event?.diffusion?.[name]
        );

        const generationConsoleStageMessage = (stage) => {
            const fallback = generationConsoleStageFallbacks[stage];
            if (!fallback) {
                return generationConsoleT(
                    "generation.console.stage.unknown",
                    {stage: generationConsoleValue(stage)},
                    "Stage: {stage}"
                );
            }
            return generationConsoleT(
                "generation.console.stage." + stage,
                {},
                fallback
            );
        };

        const generationConsoleEventMessage = (event, manifest = {}) => {
            const stage = String(event?.stage || "");
            const rawMessage = String(event?.message || "");
            const detail = (name) => generationConsoleEventDetail(event, name);
            const params = manifest.params || manifest.stats?.params || {};

            if (stage === "shape_generation" && params.steps !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.shape_generation_detail",
                    {
                        steps: generationConsoleValue(params.steps),
                        guidance: generationConsoleValue(params.guidance_scale),
                        octree: generationConsoleValue(params.octree_resolution),
                        chunks: generationConsoleValue(params.num_chunks),
                    },
                    "Launching Hunyuan3D inference \u00b7 steps={steps} guidance={guidance} octree={octree} chunks={chunks}"
                );
            }
            if (stage === "prepare_conditioning" && detail("view_count") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.prepare_conditioning_detail",
                    {view_count: generationConsoleValue(detail("view_count"))},
                    "Preparing {view_count} input view(s) for model conditioning"
                );
            }
            if (stage === "encode_conditioning" && detail("image_shape") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.encode_conditioning_detail",
                    {
                        image_shape: generationConsoleValue(detail("image_shape")),
                        dtype: generationConsoleValue(detail("dtype")),
                    },
                    "Encoding vision features from tensor {image_shape} ({dtype})"
                );
            }
            if (stage === "conditioning_ready" && detail("batch_size") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.conditioning_ready_detail",
                    {batch_size: generationConsoleValue(detail("batch_size"))},
                    "Vision conditioning ready for batch_size={batch_size}"
                );
            }
            if (
                stage === "prepare_timestep_schedule"
                && detail("requested_steps") !== undefined
            ) {
                return generationConsoleT(
                    "generation.console.stage.prepare_timestep_schedule_detail",
                    {
                        scheduler: generationConsoleValue(detail("scheduler"), "diffusion"),
                        steps: generationConsoleValue(detail("requested_steps")),
                    },
                    "Building {scheduler} schedule with {steps} steps"
                );
            }
            if (stage === "latents_initialized" && detail("latent_shape") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.latents_initialized_detail",
                    {
                        latent_shape: generationConsoleValue(detail("latent_shape")),
                        device: generationConsoleValue(detail("device"), "cuda"),
                        dtype: generationConsoleValue(detail("dtype")),
                    },
                    "Initialized latent noise {latent_shape} on {device} as {dtype}"
                );
            }
            if (stage === "diffusion_started" && detail("total_steps") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.diffusion_started_detail",
                    {total_steps: generationConsoleValue(detail("total_steps"))},
                    "Starting {total_steps} real diffusion steps on CUDA"
                );
            }
            if (
                stage === "diffusion_step"
                && detail("step") !== undefined
                && detail("total_steps") !== undefined
            ) {
                const timestep = Number(detail("timestep"));
                const stepSeconds = Number(detail("step_seconds"));
                const etaSeconds = Number(detail("eta_seconds"));
                const allocated = Number(detail("vram_allocated_gb"));
                const reserved = Number(detail("vram_reserved_gb"));
                return generationConsoleT(
                    "generation.console.stage.diffusion_step_detail",
                    {
                        step: String(detail("step")).padStart(2, "0"),
                        total_steps: String(detail("total_steps")).padStart(2, "0"),
                        timestep: Number.isFinite(timestep)
                            ? timestep.toFixed(4)
                            : generationConsoleValue(detail("timestep")),
                        step_seconds: Number.isFinite(stepSeconds)
                            ? stepSeconds.toFixed(2)
                            : generationConsoleValue(detail("step_seconds")),
                        eta_seconds: Number.isFinite(etaSeconds)
                            ? etaSeconds.toFixed(1)
                            : generationConsoleValue(detail("eta_seconds")),
                        vram_allocated: Number.isFinite(allocated)
                            ? allocated.toFixed(2)
                            : generationConsoleValue(detail("vram_allocated_gb")),
                        vram_reserved: Number.isFinite(reserved)
                            ? reserved.toFixed(2)
                            : generationConsoleValue(detail("vram_reserved_gb")),
                    },
                    "Step {step}/{total_steps} \u00b7 t={timestep} \u00b7 {step_seconds}s \u00b7 ETA {eta_seconds}s \u00b7 VRAM {vram_allocated}/{vram_reserved} GB"
                );
            }
            if (
                stage === "diffusion_completed"
                && detail("sampling_seconds") !== undefined
            ) {
                const seconds = Number(detail("sampling_seconds"));
                return generationConsoleT(
                    "generation.console.stage.diffusion_completed_detail",
                    {
                        seconds: Number.isFinite(seconds)
                            ? seconds.toFixed(2)
                            : generationConsoleValue(detail("sampling_seconds")),
                    },
                    "Diffusion sampling completed in {seconds}s"
                );
            }
            if (stage === "vae_decoding" && detail("latent_shape") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.vae_decoding_detail",
                    {
                        latent_shape: generationConsoleValue(detail("latent_shape")),
                        dtype: generationConsoleValue(detail("dtype")),
                    },
                    "ShapeVAE decoding latent {latent_shape} ({dtype})"
                );
            }
            if (stage === "volume_decoding" && detail("decoder") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.volume_decoding_detail",
                    {
                        decoder: generationConsoleValue(detail("decoder"), "volume decoder"),
                        octree: generationConsoleValue(detail("octree_resolution")),
                        chunks: generationConsoleValue(detail("num_chunks")),
                    },
                    "Starting {decoder} \u00b7 octree={octree} chunks={chunks}"
                );
            }
            if (
                stage === "volume_decoding_progress"
                && detail("chunk") !== undefined
            ) {
                const percent = Number(detail("volume_percent"));
                const etaSeconds = Number(detail("eta_seconds"));
                return generationConsoleT(
                    "generation.console.stage.volume_decoding_progress_detail",
                    {
                        chunk: generationConsoleValue(detail("chunk")),
                        total_chunks: generationConsoleValue(detail("total_chunks")),
                        processed_points: generationConsoleValue(detail("processed_points")),
                        total_points: generationConsoleValue(detail("total_points")),
                        percent: Number.isFinite(percent)
                            ? percent.toFixed(1)
                            : generationConsoleValue(detail("volume_percent")),
                        eta_seconds: Number.isFinite(etaSeconds)
                            ? etaSeconds.toFixed(1)
                            : generationConsoleValue(detail("eta_seconds")),
                    },
                    "Volume chunk {chunk}/{total_chunks} \u00b7 points {processed_points}/{total_points} \u00b7 {percent}% \u00b7 ETA {eta_seconds}s"
                );
            }
            if (
                stage === "volume_decoding_completed"
                && detail("decoder") !== undefined
            ) {
                return generationConsoleT(
                    "generation.console.stage.volume_decoding_completed_detail",
                    {
                        decoder: generationConsoleValue(detail("decoder"), "decoder"),
                        grid_shape: generationConsoleValue(detail("grid_shape")),
                    },
                    "Dense volume ready from {decoder} \u00b7 grid={grid_shape}"
                );
            }
            if (stage === "surface_extraction" && detail("extractor") !== undefined) {
                return generationConsoleT(
                    "generation.console.stage.surface_extraction_detail",
                    {extractor: generationConsoleValue(detail("extractor"))},
                    "Running marching cubes with {extractor}"
                );
            }
            if (
                stage === "surface_extraction_completed"
                && detail("mesh_count") !== undefined
            ) {
                return generationConsoleT(
                    "generation.console.stage.surface_extraction_completed_detail",
                    {mesh_count: generationConsoleValue(detail("mesh_count"))},
                    "Surface extraction completed \u00b7 mesh_count={mesh_count}"
                );
            }
            if (stage === "preparing_texture_inputs") {
                if (rawMessage.includes("ten calibrated")) {
                    return generationConsoleT(
                        "generation.console.stage.preparing_texture_inputs_ten",
                        {},
                        "Preparing all ten calibrated images for color projection"
                    );
                }
                if (rawMessage.includes("four cardinal")) {
                    return generationConsoleT(
                        "generation.console.stage.preparing_texture_inputs_four",
                        {},
                        "Preparing four cardinal images for color projection"
                    );
                }
            }
            if (stage === "baking_original") {
                if (rawMessage.includes("Top and Bottom colors")) {
                    return generationConsoleT(
                        "generation.console.stage.baking_original_six",
                        {},
                        "Projecting Front, Back, Left, Right, Top and Bottom colors onto the generated mesh"
                    );
                }
                if (rawMessage.includes("strict visibility-aware ten-view")) {
                    return generationConsoleT(
                        "generation.console.stage.baking_original_ten",
                        {},
                        "Baking strict visibility-aware ten-view color into Original GLB"
                    );
                }
                if (rawMessage.includes("visibility-aware color")) {
                    return generationConsoleT(
                        "generation.console.stage.baking_original_four",
                        {},
                        "Baking visibility-aware color into Original GLB"
                    );
                }
            }
            if (stage === "coloring_original_fallback" && rawMessage.includes("vertex colors")) {
                return generationConsoleT(
                    "generation.console.stage.coloring_original_fallback_vertex",
                    {},
                    "Using multi-view vertex colors as the Original fallback"
                );
            }
            if (stage === "scoring_original") {
                if (rawMessage.includes("Selecting the clearest")) {
                    return generationConsoleT(
                        "generation.console.stage.scoring_original_selecting",
                        {},
                        "Selecting the clearest RC-scored Original candidate"
                    );
                }
                if (rawMessage.includes("reprojection consistency")) {
                    return generationConsoleT(
                        "generation.console.stage.scoring_original_reprojection",
                        {},
                        "Scoring face and hair reprojection consistency before publication"
                    );
                }
            }
            return generationConsoleStageMessage(stage);
        };

        const generationConsoleElement = (id) => document.getElementById(id);

        const generationConsoleElapsed = (timestamp = null) => {
            const unit = generationConsoleLocale() === "zh-CN" ? "秒" : "s";
            if (!generationConsoleStartedAt) return "+00.0" + unit;
            const target = timestamp ? new Date(timestamp).getTime() : Date.now();
            const elapsed = Math.max(0, (target - generationConsoleStartedAt) / 1000);
            return "+" + elapsed.toFixed(1).padStart(4, "0") + unit;
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
            syncGenerationWorkspaceProgress(safeProgress, stage);
        };

        const setGenerationConsoleState = (state, label) => {
            const root = generationConsoleElement("generation-console");
            const status = generationConsoleElement("generation-console-status");
            if (root) root.dataset.state = state;
            if (status) status.textContent = label;
        };

        const setGenerationDetail = (id, value) => {
            const element = generationConsoleElement(id);
            if (element) element.textContent = value ?? "\u2014";
        };

        const formatGenerationCount = (value) => {
            const number = Number(value);
            if (!Number.isFinite(number)) return "\u2014";
            try {
                return new Intl.NumberFormat(generationConsoleLocale(), {
                    notation: number >= 1000 ? "compact" : "standard",
                    maximumFractionDigits: number >= 1000000 ? 2 : number >= 1000 ? 1 : 0,
                }).format(Math.round(number));
            } catch {
                return Math.round(number).toLocaleString("en-US");
            }
        };

        const formatGenerationBytes = (value) => {
            const bytes = Number(value);
            if (!Number.isFinite(bytes) || bytes <= 0) return null;
            if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
            if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
            return bytes + " B";
        };

        const resetGenerationDetails = (mode = "") => {
            setGenerationDetail("generation-info-model", "\u2014");
            setGenerationDetail(
                "generation-info-views",
                mode === "10-VIEW"
                    ? "10"
                    : mode === "6-VIEW" ? "6"
                    : mode === "4-VIEW" ? "4" : mode === "1-VIEW" ? "1" : "\u2014"
            );
            setGenerationDetail("generation-info-time", "\u2014");
            setGenerationDetail("generation-info-resolution", "\u2014");
            setGenerationDetail("generation-info-polygons", "\u2014");
            setGenerationDetail("generation-info-vertices", "\u2014");
            setGenerationDetail(
                "generation-output-meta",
                generationConsoleT("output.awaiting_mesh", {}, "Awaiting generated mesh")
            );
        };

        const updateGenerationDetails = (manifest) => {
            const params = manifest.params || manifest.stats?.params || {};
            const stats = manifest.stats || {};
            const rawModel = manifest.model?.shapegen || stats.model?.shapegen || "";
            const modelName = String(rawModel).split("/").filter(Boolean).at(-1) || "\u2014";
            const viewList = Array.isArray(params.views_provided)
                ? params.views_provided
                : params.views_used;
            const viewCount = Array.isArray(viewList)
                ? viewList.length
                : params.input_mode === "ten"
                    ? 10
                    : params.input_mode === "six"
                        ? 6
                        : params.input_mode === "four" ? 4 : params.input_mode ? 1 : "\u2014";
            const totalSeconds = Number(stats.time?.total);
            const faces = stats.number_of_faces ?? manifest.number_of_faces;
            const vertices = stats.number_of_vertices ?? manifest.number_of_vertices;

            setGenerationDetail("generation-info-model", modelName);
            setGenerationDetail("generation-info-views", String(viewCount));
            setGenerationDetail(
                "generation-info-time",
                Number.isFinite(totalSeconds)
                    ? generationConsoleT(
                        "history.duration_seconds",
                        {value: totalSeconds.toFixed(1)},
                        totalSeconds.toFixed(1) + " s"
                    )
                    : "\u2014"
            );
            setGenerationDetail("generation-info-resolution", params.octree_resolution ?? "\u2014");
            setGenerationDetail("generation-info-polygons", formatGenerationCount(faces));
            setGenerationDetail("generation-info-vertices", formatGenerationCount(vertices));

            const outputMeta = generationConsoleElement("generation-output-meta");
            if (!outputMeta) return;
            if (manifest.status !== "completed") {
                outputMeta.textContent = generationConsoleT(
                    "generation.console.output.in_progress",
                    {},
                    "Generation in progress"
                );
                return;
            }

            const uid = manifest.generation_uid;
            const meshFilename = String(manifest.outputs?.mesh || "white_mesh.glb");
            outputMeta.textContent = generationConsoleT(
                "generation.console.output.saved_source",
                {},
                "GLB \u00b7 saved to source"
            );
            fetch(
                "/static/" + encodeURIComponent(uid) + "/" + encodeURIComponent(meshFilename),
                {method: "HEAD", cache: "no-store"}
            ).then((response) => {
                if (!response.ok || generationConsoleUid !== uid) return;
                const size = formatGenerationBytes(response.headers.get("content-length"));
                if (size) {
                    outputMeta.textContent = generationConsoleT(
                        "generation.console.output.saved_size",
                        {size},
                        "GLB \u00b7 {size} \u00b7 saved"
                    );
                }
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
            generationConsoleLastManifest = manifest;

            updateGenerationDetails(manifest);

            if (
                manifest.storage_folder
                && !generationConsoleSeenEvents.has("__storage__")
            ) {
                generationConsoleSeenEvents.add("__storage__");
                appendGenerationConsoleLine(
                    generationConsoleT("generation.console.level.store", {}, "STORE"),
                    generationConsoleT(
                        "generation.console.target_path",
                        {path: manifest.storage_folder},
                        "Target: {path}"
                    ),
                    "muted"
                );
            }

            if (manifest.params && !generationConsoleParamsRendered) {
                generationConsoleParamsRendered = true;
                const params = manifest.params;
                appendGenerationConsoleLine(
                    generationConsoleT("generation.console.level.config", {}, "CONFIG"),
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
                    generationConsoleLevel(event.stage),
                    generationConsoleEventMessage(event, manifest),
                    kind,
                    event.at
                );
            });

            const lastEvent = (manifest.events || []).at(-1);
            const lastEventMessage = lastEvent
                ? generationConsoleEventMessage(lastEvent, manifest)
                : generationConsoleStageMessage(manifest.stage);
            setGenerationConsoleProgress(
                manifest.progress,
                lastEventMessage
            );
            syncGenerationWorkspaceManifest(manifest, lastEventMessage);

            const clock = generationConsoleElement("generation-console-clock");
            if (clock) {
                clock.textContent = generationConsoleT(
                    "generation.console.clock.live",
                    {elapsed: generationConsoleElapsed()},
                    "LIVE {elapsed}"
                );
            }

            if (manifest.status === "completed") {
                const stats = manifest.stats || {};
                if (!generationConsoleSeenEvents.has("__mesh_stats__")) {
                    generationConsoleSeenEvents.add("__mesh_stats__");
                    appendGenerationConsoleLine(
                        generationConsoleT("generation.console.level.stats", {}, "STATS"),
                        generationConsoleT(
                            "generation.console.mesh_stats",
                            {
                                vertices: stats.number_of_vertices ?? "-",
                                faces: stats.number_of_faces ?? "-",
                                total: Number(stats.time?.total || 0).toFixed(2),
                            },
                            "vertices={vertices} faces={faces} total={total}s"
                        ),
                        "success"
                    );
                    const storageFolder = String(
                        manifest.storage_folder || ("/static/" + generationConsoleUid)
                    ).replace(/[\/]+$/, "");
                    const meshFilename = String(manifest.outputs?.mesh || "white_mesh.glb");
                    appendGenerationConsoleLine(
                        generationConsoleT("generation.console.level.output", {}, "OUTPUT"),
                        storageFolder + "/" + meshFilename,
                        "success"
                    );
                }
                setGenerationConsoleState(
                    "completed",
                    generationConsoleT(
                        "generation.console.state.completed",
                        {},
                        "COMPLETED"
                    )
                );
                setGenerationConsoleProgress(
                    100,
                    generationConsoleT(
                        "generation.console.progress.ready",
                        {},
                        "3D model is ready"
                    )
                );
                if (clock) {
                    clock.textContent = generationConsoleT(
                        "generation.console.clock.saved_source",
                        {},
                        "SAVED TO SOURCE"
                    );
                }
                stopGenerationConsolePolling();
            } else if (manifest.status === "failed") {
                if (!generationConsoleSeenEvents.has("__error__")) {
                    generationConsoleSeenEvents.add("__error__");
                    const error = String(
                        manifest.error
                        || generationConsoleT(
                            "generation.console.error.unknown",
                            {},
                            "Unknown generation error"
                        )
                    ).replace(/^'|'$/g, "");
                    appendGenerationConsoleLine(
                        generationConsoleT("generation.console.level.error", {}, "ERROR"),
                        generationConsoleT(
                            "generation.console.error.detail",
                            {error},
                            "Generation error: {error}"
                        ),
                        "error"
                    );
                }
                setGenerationConsoleState(
                    "failed",
                    generationConsoleT("generation.console.state.failed", {}, "FAILED")
                );
                setGenerationConsoleProgress(
                    100,
                    generationConsoleT(
                        "generation.console.progress.stopped_error",
                        {},
                        "Generation stopped with an error"
                    )
                );
                if (clock) {
                    clock.textContent = generationConsoleT(
                        "generation.console.clock.error_manifest",
                        {},
                        "ERROR SAVED TO MANIFEST"
                    );
                }
                stopGenerationConsolePolling();
            } else {
                setGenerationConsoleState(
                    "running",
                    generationConsoleT("generation.console.state.running", {}, "RUNNING")
                );
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
                        appendGenerationConsoleLine(
                            generationConsoleT(
                                "generation.console.level.queue",
                                {},
                                "QUEUE"
                            ),
                            generationConsoleT(
                                "generation.console.waiting.backend",
                                {},
                                "Waiting for the backend worker..."
                            ),
                            "muted"
                        );
                        setGenerationConsoleProgress(
                            2,
                            generationConsoleT(
                                "generation.console.waiting.queue",
                                {},
                                "Waiting in the Gradio queue"
                            )
                        );
                    }
                    return;
                }
                generationConsolePollMisses = 0;
                renderGenerationManifest(await response.json());
            } catch (error) {
                generationConsolePollMisses += 1;
                if (generationConsolePollMisses === 8) {
                    appendGenerationConsoleLine(
                        generationConsoleT("generation.console.level.warn", {}, "WARN"),
                        generationConsoleT(
                            "generation.console.waiting.poll_retry",
                            {},
                            "Manifest polling will retry automatically"
                        ),
                        "muted"
                    );
                }
            }
        };

        const renderGenerationConsoleInitial = () => {
            const uid = generationConsoleUid;
            if (!uid) return;

            const log = generationConsoleElement("generation-console-log");
            if (log) log.replaceChildren();
            const job = generationConsoleElement("generation-console-job");
            if (job) job.textContent = "generation/" + uid;
            const modeElement = generationConsoleElement("generation-console-mode");
            if (modeElement) {
                modeElement.textContent = generationConsoleModeLabel(
                    generationConsoleMode
                );
            }
            const clock = generationConsoleElement("generation-console-clock");
            if (clock) {
                clock.textContent = generationConsoleT(
                    "generation.console.clock.connecting",
                    {},
                    "CONNECTING TO MANIFEST"
                );
            }
            resetGenerationDetails(generationConsoleMode);

            setGenerationConsoleState(
                "running",
                generationConsoleResumed
                    ? generationConsoleT(
                        "generation.console.state.restoring",
                        {},
                        "RESTORING"
                    )
                    : generationConsoleT(
                        "generation.console.state.starting",
                        {},
                        "STARTING"
                    )
            );
            setGenerationConsoleProgress(
                1,
                generationConsoleResumed
                    ? generationConsoleT(
                        "generation.console.progress.restoring",
                        {},
                        "Restoring generation state"
                    )
                    : generationConsoleT(
                        "generation.console.progress.dispatching",
                        {},
                        "Dispatching request"
                    )
            );
            appendGenerationConsoleLine(
                generationConsoleResumed
                    ? generationConsoleT(
                        "generation.console.level.resume",
                        {},
                        "RESUME"
                    )
                    : "$",
                (generationConsoleResumed ? "restore" : "hunyuan3d.generate")
                    + " --mode " + generationConsoleMode.toLowerCase()
                    + " --uid " + uid,
                "command"
            );
            appendGenerationConsoleLine(
                generationConsoleT("generation.console.level.store", {}, "STORE"),
                generationConsoleT(
                    "generation.console.target_waiting",
                    {},
                    "Target: waiting for generation manifest"
                ),
                "muted"
            );
        };

        const startGenerationConsole = (uid, resumed = false) => {
            if (!uid) return;

            stopGenerationConsolePolling();
            generationConsoleUid = uid;
            generationConsoleStartedAt = Date.now();
            generationConsoleSeenEvents = new Set();
            generationConsoleParamsRendered = false;
            generationConsolePollMisses = 0;
            generationConsoleLastManifest = null;
            generationConsoleResumed = resumed;
            const inputTab = currentAppUrl().searchParams.get("tab");
            const routeMode = tabRoutes.find((route) => route.slug === inputTab)?.mode
                || "single";
            generationConsoleMode = {
                single: "1-VIEW",
                four: "4-VIEW",
                six: "6-VIEW",
                ten: "10-VIEW",
            }[routeMode] || "1-VIEW";
            startGenerationWorkspaceLoading(uid, resumed);
            renderGenerationConsoleInitial();

            window.setTimeout(pollGenerationManifest, 120);
            generationConsoleTimer = window.setInterval(pollGenerationManifest, 700);
        };

        const rerenderGenerationConsoleForLocale = () => {
            if (!generationConsoleUid || !generationConsoleElement("generation-console")) return;
            generationConsoleSeenEvents = new Set();
            generationConsoleParamsRendered = false;
            renderGenerationConsoleInitial();
            if (
                generationConsoleLastManifest
                && generationConsoleLastManifest.generation_uid === generationConsoleUid
            ) {
                renderGenerationManifest(generationConsoleLastManifest);
            }
        };

        window.addEventListener(
            "ui-language-change",
            rerenderGenerationConsoleForLocale
        );

        const syncGenerationConsoleFromUrl = () => {
            const uid = currentAppUrl().searchParams.get("generation");
            if (uid && uid !== generationConsoleUid) startGenerationConsole(uid, true);
        };

        const beginGeneration = (event) => {
            const buttonRoot = document.getElementById("generate-3d-button");
            if (buttonRoot?.getAttribute("aria-busy") === "true") {
                event?.preventDefault();
                event?.stopImmediatePropagation();
                return;
            }
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
