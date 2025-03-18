export type paths = {
    "/api/v1/utilities/dynamicprompts": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Parse Dynamicprompts
         * @description Creates a batch process
         */
        post: operations["parse_dynamicprompts"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Model Records
         * @description Get a list of models.
         */
        get: operations["list_model_records"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/get_by_attrs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Records By Attrs
         * @description Gets a model by its attributes. The main use of this route is to provide backwards compatibility with the old
         *     model manager, which identified models by a combination of name, base and type.
         */
        get: operations["get_model_records_by_attrs"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/i/{key}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Record
         * @description Get a model record
         */
        get: operations["get_model_record"];
        put?: never;
        post?: never;
        /**
         * Delete Model
         * @description Delete model record from database.
         *
         *     The configuration record will be removed. The corresponding weights files will be
         *     deleted as well if they reside within the InvokeAI "models" directory.
         */
        delete: operations["delete_model"];
        options?: never;
        head?: never;
        /**
         * Update Model Record
         * @description Update a model's config.
         */
        patch: operations["update_model_record"];
        trace?: never;
    };
    "/api/v2/models/scan_folder": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Scan For Models */
        get: operations["scan_for_models"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/hugging_face": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Hugging Face Models */
        get: operations["get_hugging_face_models"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/i/{key}/image": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Image
         * @description Gets an image file that previews the model
         */
        get: operations["get_model_image"];
        put?: never;
        post?: never;
        /** Delete Model Image */
        delete: operations["delete_model_image"];
        options?: never;
        head?: never;
        /** Update Model Image */
        patch: operations["update_model_image"];
        trace?: never;
    };
    "/api/v2/models/install": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Model Installs
         * @description Return the list of model install jobs.
         *
         *     Install jobs have a numeric `id`, a `status`, and other fields that provide information on
         *     the nature of the job and its progress. The `status` is one of:
         *
         *     * "waiting" -- Job is waiting in the queue to run
         *     * "downloading" -- Model file(s) are downloading
         *     * "running" -- Model has downloaded and the model probing and registration process is running
         *     * "completed" -- Installation completed successfully
         *     * "error" -- An error occurred. Details will be in the "error_type" and "error" fields.
         *     * "cancelled" -- Job was cancelled before completion.
         *
         *     Once completed, information about the model such as its size, base
         *     model and type can be retrieved from the `config_out` field. For multi-file models such as diffusers,
         *     information on individual files can be retrieved from `download_parts`.
         *
         *     See the example and schema below for more information.
         */
        get: operations["list_model_installs"];
        put?: never;
        /**
         * Install Model
         * @description Install a model using a string identifier.
         *
         *     `source` can be any of the following.
         *
         *     1. A path on the local filesystem ('C:\users\fred\model.safetensors')
         *     2. A Url pointing to a single downloadable model file
         *     3. A HuggingFace repo_id with any of the following formats:
         *        - model/name
         *        - model/name:fp16:vae
         *        - model/name::vae          -- use default precision
         *        - model/name:fp16:path/to/model.safetensors
         *        - model/name::path/to/model.safetensors
         *
         *     `config` is a ModelRecordChanges object. Fields in this object will override
         *     the ones that are probed automatically. Pass an empty object to accept
         *     all the defaults.
         *
         *     `access_token` is an optional access token for use with Urls that require
         *     authentication.
         *
         *     Models will be downloaded, probed, configured and installed in a
         *     series of background threads. The return object has `status` attribute
         *     that can be used to monitor progress.
         *
         *     See the documentation for `import_model_record` for more information on
         *     interpreting the job information returned by this route.
         */
        post: operations["install_model"];
        /**
         * Prune Model Install Jobs
         * @description Prune all completed and errored jobs from the install job list.
         */
        delete: operations["prune_model_install_jobs"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/install/huggingface": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Install Hugging Face Model
         * @description Install a Hugging Face model using a string identifier.
         */
        get: operations["install_hugging_face_model"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/install/{id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Install Job
         * @description Return model install job corresponding to the given source. See the documentation for 'List Model Install Jobs'
         *     for information on the format of the return value.
         */
        get: operations["get_model_install_job"];
        put?: never;
        post?: never;
        /**
         * Cancel Model Install Job
         * @description Cancel the model install job(s) corresponding to the given job ID.
         */
        delete: operations["cancel_model_install_job"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/convert/{key}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Convert Model
         * @description Permanently convert a model into diffusers format, replacing the safetensors version.
         *     Note that during the conversion process the key and model hash will change.
         *     The return value is the model configuration for the converted model.
         */
        put: operations["convert_model"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/starter_models": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Starter Models */
        get: operations["get_starter_models"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/stats": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get model manager RAM cache performance statistics.
         * @description Return performance statistics on the model manager's RAM cache. Will return null if no models have been loaded.
         */
        get: operations["get_stats"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/empty_model_cache": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Empty Model Cache
         * @description Drop all models from the model cache to free RAM/VRAM. 'Locked' models that are in active use will not be dropped.
         */
        post: operations["empty_model_cache"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v2/models/hf_login": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Hf Login Status */
        get: operations["get_hf_login_status"];
        put?: never;
        /** Do Hf Login */
        post: operations["do_hf_login"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/download_queue/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Downloads
         * @description Get a list of active and inactive jobs.
         */
        get: operations["list_downloads"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /**
         * Prune Downloads
         * @description Prune completed and errored jobs.
         */
        patch: operations["prune_downloads"];
        trace?: never;
    };
    "/api/v1/download_queue/i/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Download
         * @description Download the source URL to the file or directory indicted in dest.
         */
        post: operations["download"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/download_queue/i/{id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Download Job
         * @description Get a download job using its ID.
         */
        get: operations["get_download_job"];
        put?: never;
        post?: never;
        /**
         * Cancel Download Job
         * @description Cancel a download job using its ID.
         */
        delete: operations["cancel_download_job"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/download_queue/i": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /**
         * Cancel All Download Jobs
         * @description Cancel all download jobs.
         */
        delete: operations["cancel_all_download_jobs"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/upload": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Upload Image
         * @description Uploads an image
         */
        post: operations["upload_image"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Image Dtos
         * @description Gets a list of image DTOs
         */
        get: operations["list_image_dtos"];
        put?: never;
        /**
         * Create Image Upload Entry
         * @description Uploads an image from a URL, not implemented
         */
        post: operations["create_image_upload_entry"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/i/{image_name}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Image Dto
         * @description Gets an image's DTO
         */
        get: operations["get_image_dto"];
        put?: never;
        post?: never;
        /**
         * Delete Image
         * @description Deletes an image
         */
        delete: operations["delete_image"];
        options?: never;
        head?: never;
        /**
         * Update Image
         * @description Updates an image
         */
        patch: operations["update_image"];
        trace?: never;
    };
    "/api/v1/images/intermediates": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Intermediates Count
         * @description Gets the count of intermediate images
         */
        get: operations["get_intermediates_count"];
        put?: never;
        post?: never;
        /**
         * Clear Intermediates
         * @description Clears all intermediates
         */
        delete: operations["clear_intermediates"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/i/{image_name}/metadata": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Image Metadata
         * @description Gets an image's metadata
         */
        get: operations["get_image_metadata"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/i/{image_name}/workflow": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Image Workflow */
        get: operations["get_image_workflow"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/i/{image_name}/full": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Image Full
         * @description Gets a full-resolution image file
         */
        get: operations["get_image_full"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        /**
         * Get Image Full
         * @description Gets a full-resolution image file
         */
        head: operations["get_image_full_head"];
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/i/{image_name}/thumbnail": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Image Thumbnail
         * @description Gets a thumbnail image file
         */
        get: operations["get_image_thumbnail"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/i/{image_name}/urls": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Image Urls
         * @description Gets an image and thumbnail URL
         */
        get: operations["get_image_urls"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/delete": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Delete Images From List */
        post: operations["delete_images_from_list"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/star": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Star Images In List */
        post: operations["star_images_in_list"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/unstar": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Unstar Images In List */
        post: operations["unstar_images_in_list"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/download": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Download Images From List */
        post: operations["download_images_from_list"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/images/download/{bulk_download_item_name}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Bulk Download Item
         * @description Gets a bulk download zip file
         */
        get: operations["get_bulk_download_item"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/boards/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Boards
         * @description Gets a list of boards
         */
        get: operations["list_boards"];
        put?: never;
        /**
         * Create Board
         * @description Creates a board
         */
        post: operations["create_board"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/boards/{board_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Board
         * @description Gets a board
         */
        get: operations["get_board"];
        put?: never;
        post?: never;
        /**
         * Delete Board
         * @description Deletes a board
         */
        delete: operations["delete_board"];
        options?: never;
        head?: never;
        /**
         * Update Board
         * @description Updates a board
         */
        patch: operations["update_board"];
        trace?: never;
    };
    "/api/v1/boards/{board_id}/image_names": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List All Board Image Names
         * @description Gets a list of images for a board
         */
        get: operations["list_all_board_image_names"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/board_images/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Add Image To Board
         * @description Creates a board_image
         */
        post: operations["add_image_to_board"];
        /**
         * Remove Image From Board
         * @description Removes an image from its board, if it had one
         */
        delete: operations["remove_image_from_board"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/board_images/batch": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Add Images To Board
         * @description Adds a list of images to a board
         */
        post: operations["add_images_to_board"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/board_images/batch/delete": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Remove Images From Board
         * @description Removes a list of images from their board, if they had one
         */
        post: operations["remove_images_from_board"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/version": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Version */
        get: operations["app_version"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/app_deps": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get App Deps */
        get: operations["get_app_deps"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/config": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Config  */
        get: operations["get_config"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/runtime_config": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Get Runtime Config */
        get: operations["get_runtime_config"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/logging": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Log Level
         * @description Returns the log level
         */
        get: operations["get_log_level"];
        put?: never;
        /**
         * Set Log Level
         * @description Sets the log verbosity level
         */
        post: operations["set_log_level"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/invocation_cache": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        /**
         * Clear Invocation Cache
         * @description Clears the invocation cache
         */
        delete: operations["clear_invocation_cache"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/invocation_cache/enable": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Enable Invocation Cache
         * @description Clears the invocation cache
         */
        put: operations["enable_invocation_cache"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/invocation_cache/disable": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Disable Invocation Cache
         * @description Clears the invocation cache
         */
        put: operations["disable_invocation_cache"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/app/invocation_cache/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Invocation Cache Status
         * @description Clears the invocation cache
         */
        get: operations["get_invocation_cache_status"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/enqueue_batch": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Enqueue Batch
         * @description Processes a batch and enqueues the output graphs for execution.
         */
        post: operations["enqueue_batch"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/list": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Queue Items
         * @description Gets all queue items (without graphs)
         */
        get: operations["list_queue_items"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/processor/resume": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Resume
         * @description Resumes session processor
         */
        put: operations["resume"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/processor/pause": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Pause
         * @description Pauses session processor
         */
        put: operations["pause"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/cancel_all_except_current": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Cancel All Except Current
         * @description Immediately cancels all queue items except in-processing items
         */
        put: operations["cancel_all_except_current"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/cancel_by_batch_ids": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Cancel By Batch Ids
         * @description Immediately cancels all queue items from the given batch ids
         */
        put: operations["cancel_by_batch_ids"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/cancel_by_destination": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Cancel By Destination
         * @description Immediately cancels all queue items with the given origin
         */
        put: operations["cancel_by_destination"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/retry_items_by_id": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Retry Items By Id
         * @description Immediately cancels all queue items with the given origin
         */
        put: operations["retry_items_by_id"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/clear": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Clear
         * @description Clears the queue entirely, immediately canceling the currently-executing session
         */
        put: operations["clear"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/prune": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Prune
         * @description Prunes all completed or errored queue items
         */
        put: operations["prune"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/current": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Current Queue Item
         * @description Gets the currently execution queue item
         */
        get: operations["get_current_queue_item"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/next": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Next Queue Item
         * @description Gets the next queue item, without executing it
         */
        get: operations["get_next_queue_item"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Queue Status
         * @description Gets the status of the session queue
         */
        get: operations["get_queue_status"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/b/{batch_id}/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Batch Status
         * @description Gets the status of the session queue
         */
        get: operations["get_batch_status"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/i/{item_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Queue Item
         * @description Gets a queue item
         */
        get: operations["get_queue_item"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/i/{item_id}/cancel": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Cancel Queue Item
         * @description Deletes a queue item
         */
        put: operations["cancel_queue_item"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/queue/{queue_id}/counts_by_destination": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Counts By Destination
         * @description Gets the counts of queue items by destination
         */
        get: operations["counts_by_destination"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/workflows/i/{workflow_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Workflow
         * @description Gets a workflow
         */
        get: operations["get_workflow"];
        put?: never;
        post?: never;
        /**
         * Delete Workflow
         * @description Deletes a workflow
         */
        delete: operations["delete_workflow"];
        options?: never;
        head?: never;
        /**
         * Update Workflow
         * @description Updates a workflow
         */
        patch: operations["update_workflow"];
        trace?: never;
    };
    "/api/v1/workflows/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Workflows
         * @description Gets a page of workflows
         */
        get: operations["list_workflows"];
        put?: never;
        /**
         * Create Workflow
         * @description Creates a workflow
         */
        post: operations["create_workflow"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/workflows/i/{workflow_id}/thumbnail": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Workflow Thumbnail
         * @description Gets a workflow's thumbnail image
         */
        get: operations["get_workflow_thumbnail"];
        /**
         * Set Workflow Thumbnail
         * @description Sets a workflow's thumbnail image
         */
        put: operations["set_workflow_thumbnail"];
        post?: never;
        /**
         * Delete Workflow Thumbnail
         * @description Removes a workflow's thumbnail image
         */
        delete: operations["delete_workflow_thumbnail"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/workflows/counts_by_tag": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Counts By Tag
         * @description Counts workflows by tag
         */
        get: operations["get_counts_by_tag"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/workflows/counts_by_category": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Counts By Category
         * @description Counts workflows by category
         */
        get: operations["counts_by_category"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/workflows/i/{workflow_id}/opened_at": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        /**
         * Update Opened At
         * @description Updates the opened_at field of a workflow
         */
        put: operations["update_opened_at"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/style_presets/i/{style_preset_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Style Preset
         * @description Gets a style preset
         */
        get: operations["get_style_preset"];
        put?: never;
        post?: never;
        /**
         * Delete Style Preset
         * @description Deletes a style preset
         */
        delete: operations["delete_style_preset"];
        options?: never;
        head?: never;
        /**
         * Update Style Preset
         * @description Updates a style preset
         */
        patch: operations["update_style_preset"];
        trace?: never;
    };
    "/api/v1/style_presets/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Style Presets
         * @description Gets a page of style presets
         */
        get: operations["list_style_presets"];
        put?: never;
        /**
         * Create Style Preset
         * @description Creates a style preset
         */
        post: operations["create_style_preset"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/style_presets/i/{style_preset_id}/image": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Style Preset Image
         * @description Gets an image file that previews the model
         */
        get: operations["get_style_preset_image"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/style_presets/export": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Export Style Presets */
        get: operations["export_style_presets"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/v1/style_presets/import": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Import Style Presets */
        post: operations["import_style_presets"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
};
export type webhooks = Record<string, never>;
export type components = {
    schemas: {
        /** AddImagesToBoardResult */
        AddImagesToBoardResult: {
            /**
             * Board Id
             * @description The id of the board the images were added to
             */
            board_id: string;
            /**
             * Added Image Names
             * @description The image names that were added to the board
             */
            added_image_names: string[];
        };
        /**
         * Add Integers
         * @description Adds two numbers
         */
        AddInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * A
             * @description The first number
             * @default 0
             */
            a?: number;
            /**
             * B
             * @description The second number
             * @default 0
             */
            b?: number;
            /**
             * type
             * @default add
             * @constant
             */
            type: "add";
        };
        /**
         * Alpha Mask to Tensor
         * @description Convert a mask image to a tensor. Opaque regions are 1 and transparent regions are 0.
         */
        AlphaMaskToTensorInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The mask image to convert.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Invert
             * @description Whether to invert the mask.
             * @default false
             */
            invert?: boolean;
            /**
             * type
             * @default alpha_mask_to_tensor
             * @constant
             */
            type: "alpha_mask_to_tensor";
        };
        /**
         * AppConfig
         * @description App Config Response
         */
        AppConfig: {
            /**
             * Infill Methods
             * @description List of available infill methods
             */
            infill_methods: string[];
            /**
             * Upscaling Methods
             * @description List of upscaling methods
             */
            upscaling_methods: components["schemas"]["Upscaler"][];
            /**
             * Nsfw Methods
             * @description List of NSFW checking methods
             */
            nsfw_methods: string[];
            /**
             * Watermarking Methods
             * @description List of invisible watermark methods
             */
            watermarking_methods: string[];
        };
        /**
         * AppDependencyVersions
         * @description App depencency Versions Response
         */
        AppDependencyVersions: {
            /**
             * Accelerate
             * @description accelerate version
             */
            accelerate: string;
            /**
             * Compel
             * @description compel version
             */
            compel: string;
            /**
             * Cuda
             * @description CUDA version
             */
            cuda: string | null;
            /**
             * Diffusers
             * @description diffusers version
             */
            diffusers: string;
            /**
             * Numpy
             * @description Numpy version
             */
            numpy: string;
            /**
             * Opencv
             * @description OpenCV version
             */
            opencv: string;
            /**
             * Onnx
             * @description ONNX version
             */
            onnx: string;
            /**
             * Pillow
             * @description Pillow (PIL) version
             */
            pillow: string;
            /**
             * Python
             * @description Python version
             */
            python: string;
            /**
             * Torch
             * @description PyTorch version
             */
            torch: string;
            /**
             * Torchvision
             * @description PyTorch Vision version
             */
            torchvision: string;
            /**
             * Transformers
             * @description transformers version
             */
            transformers: string;
            /**
             * Xformers
             * @description xformers version
             */
            xformers: string | null;
        };
        /**
         * AppVersion
         * @description App Version Response
         */
        AppVersion: {
            /**
             * Version
             * @description App version
             */
            version: string;
            /**
             * Highlights
             * @description Highlights of release
             */
            highlights?: string[] | null;
        };
        /**
         * Apply Tensor Mask to Image
         * @description Applies a tensor mask to an image.
         *
         *     The image is converted to RGBA and the mask is applied to the alpha channel.
         */
        ApplyMaskTensorToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The mask tensor to apply.
             * @default null
             */
            mask?: components["schemas"]["TensorField"];
            /**
             * @description The image to apply the mask to.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Invert
             * @description Whether to invert the mask.
             * @default false
             */
            invert?: boolean;
            /**
             * type
             * @default apply_tensor_mask_to_image
             * @constant
             */
            type: "apply_tensor_mask_to_image";
        };
        /**
         * Apply Mask to Image
         * @description Extracts a region from a generated image using a mask and blends it seamlessly onto a source image.
         *     The mask uses black to indicate areas to keep from the generated image and white for areas to discard.
         */
        ApplyMaskToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image from which to extract the masked region
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The mask defining the region (black=keep, white=discard)
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * Invert Mask
             * @description Whether to invert the mask before applying it
             * @default false
             */
            invert_mask?: boolean;
            /**
             * type
             * @default apply_mask_to_image
             * @constant
             */
            type: "apply_mask_to_image";
        };
        /**
         * BaseMetadata
         * @description Adds typing data for discriminated union.
         */
        BaseMetadata: {
            /**
             * Name
             * @description model's name
             */
            name: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "basemetadata";
        };
        /**
         * BaseModelType
         * @description Base model type.
         * @enum {string}
         */
        BaseModelType: "any" | "sd-1" | "sd-2" | "sd-3" | "sdxl" | "sdxl-refiner" | "flux" | "cogview4";
        /** Batch */
        Batch: {
            /**
             * Batch Id
             * @description The ID of the batch
             */
            batch_id?: string;
            /**
             * Origin
             * @description The origin of this queue item. This data is used by the frontend to determine how to handle results.
             */
            origin?: string | null;
            /**
             * Destination
             * @description The origin of this queue item. This data is used by the frontend to determine how to handle results
             */
            destination?: string | null;
            /**
             * Data
             * @description The batch data collection.
             */
            data?: components["schemas"]["BatchDatum"][][] | null;
            /** @description The graph to initialize the session with */
            graph: components["schemas"]["Graph"];
            /** @description The workflow to initialize the session with */
            workflow?: components["schemas"]["WorkflowWithoutID"] | null;
            /**
             * Runs
             * @description Int stating how many times to iterate through all possible batch indices
             * @default 1
             */
            runs: number;
        };
        /** BatchDatum */
        BatchDatum: {
            /**
             * Node Path
             * @description The node into which this batch data collection will be substituted.
             */
            node_path: string;
            /**
             * Field Name
             * @description The field into which this batch data collection will be substituted.
             */
            field_name: string;
            /**
             * Items
             * @description The list of items to substitute into the node/field.
             */
            items?: (string | number | components["schemas"]["ImageField"])[];
        };
        /**
         * BatchEnqueuedEvent
         * @description Event model for batch_enqueued
         */
        BatchEnqueuedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Batch Id
             * @description The ID of the batch
             */
            batch_id: string;
            /**
             * Enqueued
             * @description The number of invocations enqueued
             */
            enqueued: number;
            /**
             * Requested
             * @description The number of invocations initially requested to be enqueued (may be less than enqueued if queue was full)
             */
            requested: number;
            /**
             * Priority
             * @description The priority of the batch
             */
            priority: number;
            /**
             * Origin
             * @description The origin of the batch
             * @default null
             */
            origin: string | null;
        };
        /** BatchStatus */
        BatchStatus: {
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Batch Id
             * @description The ID of the batch
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of the batch
             */
            origin: string | null;
            /**
             * Destination
             * @description The destination of the batch
             */
            destination: string | null;
            /**
             * Pending
             * @description Number of queue items with status 'pending'
             */
            pending: number;
            /**
             * In Progress
             * @description Number of queue items with status 'in_progress'
             */
            in_progress: number;
            /**
             * Completed
             * @description Number of queue items with status 'complete'
             */
            completed: number;
            /**
             * Failed
             * @description Number of queue items with status 'error'
             */
            failed: number;
            /**
             * Canceled
             * @description Number of queue items with status 'canceled'
             */
            canceled: number;
            /**
             * Total
             * @description Total number of queue items
             */
            total: number;
        };
        /**
         * Blank Image
         * @description Creates a blank image and forwards it to the pipeline
         */
        BlankImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Width
             * @description The width of the image
             * @default 512
             */
            width?: number;
            /**
             * Height
             * @description The height of the image
             * @default 512
             */
            height?: number;
            /**
             * Mode
             * @description The mode of the image
             * @default RGB
             * @enum {string}
             */
            mode?: "RGB" | "RGBA";
            /**
             * @description The color of the image
             * @default {
             *       "r": 0,
             *       "g": 0,
             *       "b": 0,
             *       "a": 255
             *     }
             */
            color?: components["schemas"]["ColorField"];
            /**
             * type
             * @default blank_image
             * @constant
             */
            type: "blank_image";
        };
        /**
         * Blend Latents
         * @description Blend two latents using a given alpha. If a mask is provided, the second latents will be masked before blending.
         *     Latents must have same size. Masking functionality added by @dwringer.
         */
        BlendLatentsInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents_a?: components["schemas"]["LatentsField"];
            /**
             * @description Latents tensor
             * @default null
             */
            latents_b?: components["schemas"]["LatentsField"];
            /**
             * @description Mask for blending in latents B
             * @default null
             */
            mask?: components["schemas"]["ImageField"] | null;
            /**
             * Alpha
             * @description Blending factor. 0.0 = use input A only, 1.0 = use input B only, 0.5 = 50% mix of input A and input B.
             * @default 0.5
             */
            alpha?: number;
            /**
             * type
             * @default lblend
             * @constant
             */
            type: "lblend";
        };
        /** BoardChanges */
        BoardChanges: {
            /**
             * Board Name
             * @description The board's new name.
             */
            board_name?: string | null;
            /**
             * Cover Image Name
             * @description The name of the board's new cover image.
             */
            cover_image_name?: string | null;
            /**
             * Archived
             * @description Whether or not the board is archived
             */
            archived?: boolean | null;
        };
        /**
         * BoardDTO
         * @description Deserialized board record with cover image URL and image count.
         */
        BoardDTO: {
            /**
             * Board Id
             * @description The unique ID of the board.
             */
            board_id: string;
            /**
             * Board Name
             * @description The name of the board.
             */
            board_name: string;
            /**
             * Created At
             * @description The created timestamp of the board.
             */
            created_at: string;
            /**
             * Updated At
             * @description The updated timestamp of the board.
             */
            updated_at: string;
            /**
             * Deleted At
             * @description The deleted timestamp of the board.
             */
            deleted_at?: string | null;
            /**
             * Cover Image Name
             * @description The name of the board's cover image.
             */
            cover_image_name: string | null;
            /**
             * Archived
             * @description Whether or not the board is archived.
             */
            archived: boolean;
            /**
             * Is Private
             * @description Whether the board is private.
             */
            is_private?: boolean | null;
            /**
             * Image Count
             * @description The number of images in the board.
             */
            image_count: number;
        };
        /**
         * BoardField
         * @description A board primitive field
         */
        BoardField: {
            /**
             * Board Id
             * @description The id of the board
             */
            board_id: string;
        };
        /**
         * BoardRecordOrderBy
         * @description The order by options for board records
         * @enum {string}
         */
        BoardRecordOrderBy: "created_at" | "board_name";
        /** Body_add_image_to_board */
        Body_add_image_to_board: {
            /**
             * Board Id
             * @description The id of the board to add to
             */
            board_id: string;
            /**
             * Image Name
             * @description The name of the image to add
             */
            image_name: string;
        };
        /** Body_add_images_to_board */
        Body_add_images_to_board: {
            /**
             * Board Id
             * @description The id of the board to add to
             */
            board_id: string;
            /**
             * Image Names
             * @description The names of the images to add
             */
            image_names: string[];
        };
        /** Body_cancel_by_batch_ids */
        Body_cancel_by_batch_ids: {
            /**
             * Batch Ids
             * @description The list of batch_ids to cancel all queue items for
             */
            batch_ids: string[];
        };
        /** Body_create_image_upload_entry */
        Body_create_image_upload_entry: {
            /**
             * Width
             * @description The width of the image
             */
            width: number;
            /**
             * Height
             * @description The height of the image
             */
            height: number;
            /**
             * Board Id
             * @description The board to add this image to, if any
             */
            board_id?: string | null;
        };
        /** Body_create_style_preset */
        Body_create_style_preset: {
            /**
             * Image
             * @description The image file to upload
             */
            image?: Blob | null;
            /**
             * Data
             * @description The data of the style preset to create
             */
            data: string;
        };
        /** Body_create_workflow */
        Body_create_workflow: {
            /** @description The workflow to create */
            workflow: components["schemas"]["WorkflowWithoutID"];
        };
        /** Body_delete_images_from_list */
        Body_delete_images_from_list: {
            /**
             * Image Names
             * @description The list of names of images to delete
             */
            image_names: string[];
        };
        /** Body_do_hf_login */
        Body_do_hf_login: {
            /**
             * Token
             * @description Hugging Face token to use for login
             */
            token: string;
        };
        /** Body_download */
        Body_download: {
            /**
             * Source
             * Format: uri
             * @description download source
             */
            source: string;
            /**
             * Dest
             * @description download destination
             */
            dest: string;
            /**
             * Priority
             * @description queue priority
             * @default 10
             */
            priority?: number;
            /**
             * Access Token
             * @description token for authorization to download
             */
            access_token?: string | null;
        };
        /** Body_download_images_from_list */
        Body_download_images_from_list: {
            /**
             * Image Names
             * @description The list of names of images to download
             */
            image_names?: string[] | null;
            /**
             * Board Id
             * @description The board from which image should be downloaded
             */
            board_id?: string | null;
        };
        /** Body_enqueue_batch */
        Body_enqueue_batch: {
            /** @description Batch to process */
            batch: components["schemas"]["Batch"];
            /**
             * Prepend
             * @description Whether or not to prepend this batch in the queue
             * @default false
             */
            prepend?: boolean;
            /** @description The validation run data to use for this batch. This is only used if this is a validation run. */
            validation_run_data?: components["schemas"]["ValidationRunData"] | null;
        };
        /** Body_import_style_presets */
        Body_import_style_presets: {
            /**
             * File
             * Format: binary
             * @description The file to import
             */
            file: Blob;
        };
        /** Body_parse_dynamicprompts */
        Body_parse_dynamicprompts: {
            /**
             * Prompt
             * @description The prompt to parse with dynamicprompts
             */
            prompt: string;
            /**
             * Max Prompts
             * @description The max number of prompts to generate
             * @default 1000
             */
            max_prompts?: number;
            /**
             * Combinatorial
             * @description Whether to use the combinatorial generator
             * @default true
             */
            combinatorial?: boolean;
            /**
             * Seed
             * @description The seed to use for random generation. Only used if not combinatorial
             */
            seed?: number | null;
        };
        /** Body_remove_image_from_board */
        Body_remove_image_from_board: {
            /**
             * Image Name
             * @description The name of the image to remove
             */
            image_name: string;
        };
        /** Body_remove_images_from_board */
        Body_remove_images_from_board: {
            /**
             * Image Names
             * @description The names of the images to remove
             */
            image_names: string[];
        };
        /** Body_set_workflow_thumbnail */
        Body_set_workflow_thumbnail: {
            /**
             * Image
             * Format: binary
             * @description The image file to upload
             */
            image: Blob;
        };
        /** Body_star_images_in_list */
        Body_star_images_in_list: {
            /**
             * Image Names
             * @description The list of names of images to star
             */
            image_names: string[];
        };
        /** Body_unstar_images_in_list */
        Body_unstar_images_in_list: {
            /**
             * Image Names
             * @description The list of names of images to unstar
             */
            image_names: string[];
        };
        /** Body_update_model_image */
        Body_update_model_image: {
            /**
             * Image
             * Format: binary
             */
            image: Blob;
        };
        /** Body_update_style_preset */
        Body_update_style_preset: {
            /**
             * Image
             * @description The image file to upload
             */
            image?: Blob | null;
            /**
             * Data
             * @description The data of the style preset to update
             */
            data: string;
        };
        /** Body_update_workflow */
        Body_update_workflow: {
            /** @description The updated workflow */
            workflow: components["schemas"]["Workflow"];
        };
        /** Body_upload_image */
        Body_upload_image: {
            /**
             * File
             * Format: binary
             */
            file: Blob;
            /**
             * Metadata
             * @description The metadata to associate with the image, must be a stringified JSON dict
             */
            metadata?: string | null;
        };
        /**
         * Boolean Collection Primitive
         * @description A collection of boolean primitive values
         */
        BooleanCollectionInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The collection of boolean values
             * @default []
             */
            collection?: boolean[];
            /**
             * type
             * @default boolean_collection
             * @constant
             */
            type: "boolean_collection";
        };
        /**
         * BooleanCollectionOutput
         * @description Base class for nodes that output a collection of booleans
         */
        BooleanCollectionOutput: {
            /**
             * Collection
             * @description The output boolean collection
             */
            collection: boolean[];
            /**
             * type
             * @default boolean_collection_output
             * @constant
             */
            type: "boolean_collection_output";
        };
        /**
         * Boolean Primitive
         * @description A boolean primitive value
         */
        BooleanInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Value
             * @description The boolean value
             * @default false
             */
            value?: boolean;
            /**
             * type
             * @default boolean
             * @constant
             */
            type: "boolean";
        };
        /**
         * BooleanOutput
         * @description Base class for nodes that output a single boolean
         */
        BooleanOutput: {
            /**
             * Value
             * @description The output boolean
             */
            value: boolean;
            /**
             * type
             * @default boolean_output
             * @constant
             */
            type: "boolean_output";
        };
        /**
         * BoundingBoxCollectionOutput
         * @description Base class for nodes that output a collection of bounding boxes
         */
        BoundingBoxCollectionOutput: {
            /**
             * Bounding Boxes
             * @description The output bounding boxes.
             */
            collection: components["schemas"]["BoundingBoxField"][];
            /**
             * type
             * @default bounding_box_collection_output
             * @constant
             */
            type: "bounding_box_collection_output";
        };
        /**
         * BoundingBoxField
         * @description A bounding box primitive value.
         */
        BoundingBoxField: {
            /**
             * X Min
             * @description The minimum x-coordinate of the bounding box (inclusive).
             */
            x_min: number;
            /**
             * X Max
             * @description The maximum x-coordinate of the bounding box (exclusive).
             */
            x_max: number;
            /**
             * Y Min
             * @description The minimum y-coordinate of the bounding box (inclusive).
             */
            y_min: number;
            /**
             * Y Max
             * @description The maximum y-coordinate of the bounding box (exclusive).
             */
            y_max: number;
            /**
             * Score
             * @description The score associated with the bounding box. In the range [0, 1]. This value is typically set when the bounding box was produced by a detector and has an associated confidence score.
             * @default null
             */
            score?: number | null;
        };
        /**
         * Bounding Box
         * @description Create a bounding box manually by supplying box coordinates
         */
        BoundingBoxInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * X Min
             * @description x-coordinate of the bounding box's top left vertex
             * @default 0
             */
            x_min?: number;
            /**
             * Y Min
             * @description y-coordinate of the bounding box's top left vertex
             * @default 0
             */
            y_min?: number;
            /**
             * X Max
             * @description x-coordinate of the bounding box's bottom right vertex
             * @default 0
             */
            x_max?: number;
            /**
             * Y Max
             * @description y-coordinate of the bounding box's bottom right vertex
             * @default 0
             */
            y_max?: number;
            /**
             * type
             * @default bounding_box
             * @constant
             */
            type: "bounding_box";
        };
        /**
         * BoundingBoxOutput
         * @description Base class for nodes that output a single bounding box
         */
        BoundingBoxOutput: {
            /** @description The output bounding box. */
            bounding_box: components["schemas"]["BoundingBoxField"];
            /**
             * type
             * @default bounding_box_output
             * @constant
             */
            type: "bounding_box_output";
        };
        /**
         * BulkDownloadCompleteEvent
         * @description Event model for bulk_download_complete
         */
        BulkDownloadCompleteEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Bulk Download Id
             * @description The ID of the bulk image download
             */
            bulk_download_id: string;
            /**
             * Bulk Download Item Id
             * @description The ID of the bulk image download item
             */
            bulk_download_item_id: string;
            /**
             * Bulk Download Item Name
             * @description The name of the bulk image download item
             */
            bulk_download_item_name: string;
        };
        /**
         * BulkDownloadErrorEvent
         * @description Event model for bulk_download_error
         */
        BulkDownloadErrorEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Bulk Download Id
             * @description The ID of the bulk image download
             */
            bulk_download_id: string;
            /**
             * Bulk Download Item Id
             * @description The ID of the bulk image download item
             */
            bulk_download_item_id: string;
            /**
             * Bulk Download Item Name
             * @description The name of the bulk image download item
             */
            bulk_download_item_name: string;
            /**
             * Error
             * @description The error message
             */
            error: string;
        };
        /**
         * BulkDownloadStartedEvent
         * @description Event model for bulk_download_started
         */
        BulkDownloadStartedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Bulk Download Id
             * @description The ID of the bulk image download
             */
            bulk_download_id: string;
            /**
             * Bulk Download Item Id
             * @description The ID of the bulk image download item
             */
            bulk_download_item_id: string;
            /**
             * Bulk Download Item Name
             * @description The name of the bulk image download item
             */
            bulk_download_item_name: string;
        };
        /** CLIPField */
        CLIPField: {
            /** @description Info to load tokenizer submodel */
            tokenizer: components["schemas"]["ModelIdentifierField"];
            /** @description Info to load text_encoder submodel */
            text_encoder: components["schemas"]["ModelIdentifierField"];
            /**
             * Skipped Layers
             * @description Number of skipped layers in text_encoder
             */
            skipped_layers: number;
            /**
             * Loras
             * @description LoRAs to apply on model loading
             */
            loras: components["schemas"]["LoRAField"][];
        };
        /**
         * CLIPGEmbedDiffusersConfig
         * @description Model config for CLIP-G Embeddings.
         */
        CLIPGEmbedDiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default clip_embed
             * @constant
             */
            type: "clip_embed";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
            /**
             * Variant
             * @default gigantic
             * @constant
             */
            variant?: "gigantic";
        };
        /**
         * CLIPLEmbedDiffusersConfig
         * @description Model config for CLIP-L Embeddings.
         */
        CLIPLEmbedDiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default clip_embed
             * @constant
             */
            type: "clip_embed";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
            /**
             * Variant
             * @default large
             * @constant
             */
            variant?: "large";
        };
        /**
         * CLIPOutput
         * @description Base class for invocations that output a CLIP field
         */
        CLIPOutput: {
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip: components["schemas"]["CLIPField"];
            /**
             * type
             * @default clip_output
             * @constant
             */
            type: "clip_output";
        };
        /**
         * Apply CLIP Skip - SD1.5, SDXL
         * @description Skip layers in clip text_encoder model.
         */
        CLIPSkipInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"];
            /**
             * Skipped Layers
             * @description Number of layers to skip in text encoder
             * @default 0
             */
            skipped_layers?: number;
            /**
             * type
             * @default clip_skip
             * @constant
             */
            type: "clip_skip";
        };
        /**
         * CLIPSkipInvocationOutput
         * @description CLIP skip node output
         */
        CLIPSkipInvocationOutput: {
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default clip_skip_output
             * @constant
             */
            type: "clip_skip_output";
        };
        /**
         * CLIPVisionDiffusersConfig
         * @description Model config for CLIPVision.
         */
        CLIPVisionDiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default clip_vision
             * @constant
             */
            type: "clip_vision";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
        };
        /**
         * CV2 Infill
         * @description Infills transparent areas of an image using OpenCV Inpainting
         */
        CV2InfillInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default infill_cv2
             * @constant
             */
            type: "infill_cv2";
        };
        /** CacheStats */
        CacheStats: {
            /**
             * Hits
             * @default 0
             */
            hits?: number;
            /**
             * Misses
             * @default 0
             */
            misses?: number;
            /**
             * High Watermark
             * @default 0
             */
            high_watermark?: number;
            /**
             * In Cache
             * @default 0
             */
            in_cache?: number;
            /**
             * Cleared
             * @default 0
             */
            cleared?: number;
            /**
             * Cache Size
             * @default 0
             */
            cache_size?: number;
            /** Loaded Model Sizes */
            loaded_model_sizes?: {
                [key: string]: number;
            };
        };
        /**
         * Calculate Image Tiles Even Split
         * @description Calculate the coordinates and overlaps of tiles that cover a target image shape.
         */
        CalculateImageTilesEvenSplitInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Image Width
             * @description The image width, in pixels, to calculate tiles for.
             * @default 1024
             */
            image_width?: number;
            /**
             * Image Height
             * @description The image height, in pixels, to calculate tiles for.
             * @default 1024
             */
            image_height?: number;
            /**
             * Num Tiles X
             * @description Number of tiles to divide image into on the x axis
             * @default 2
             */
            num_tiles_x?: number;
            /**
             * Num Tiles Y
             * @description Number of tiles to divide image into on the y axis
             * @default 2
             */
            num_tiles_y?: number;
            /**
             * Overlap
             * @description The overlap, in pixels, between adjacent tiles.
             * @default 128
             */
            overlap?: number;
            /**
             * type
             * @default calculate_image_tiles_even_split
             * @constant
             */
            type: "calculate_image_tiles_even_split";
        };
        /**
         * Calculate Image Tiles
         * @description Calculate the coordinates and overlaps of tiles that cover a target image shape.
         */
        CalculateImageTilesInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Image Width
             * @description The image width, in pixels, to calculate tiles for.
             * @default 1024
             */
            image_width?: number;
            /**
             * Image Height
             * @description The image height, in pixels, to calculate tiles for.
             * @default 1024
             */
            image_height?: number;
            /**
             * Tile Width
             * @description The tile width, in pixels.
             * @default 576
             */
            tile_width?: number;
            /**
             * Tile Height
             * @description The tile height, in pixels.
             * @default 576
             */
            tile_height?: number;
            /**
             * Overlap
             * @description The target overlap, in pixels, between adjacent tiles. Adjacent tiles will overlap by at least this amount
             * @default 128
             */
            overlap?: number;
            /**
             * type
             * @default calculate_image_tiles
             * @constant
             */
            type: "calculate_image_tiles";
        };
        /**
         * Calculate Image Tiles Minimum Overlap
         * @description Calculate the coordinates and overlaps of tiles that cover a target image shape.
         */
        CalculateImageTilesMinimumOverlapInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Image Width
             * @description The image width, in pixels, to calculate tiles for.
             * @default 1024
             */
            image_width?: number;
            /**
             * Image Height
             * @description The image height, in pixels, to calculate tiles for.
             * @default 1024
             */
            image_height?: number;
            /**
             * Tile Width
             * @description The tile width, in pixels.
             * @default 576
             */
            tile_width?: number;
            /**
             * Tile Height
             * @description The tile height, in pixels.
             * @default 576
             */
            tile_height?: number;
            /**
             * Min Overlap
             * @description Minimum overlap between adjacent tiles, in pixels.
             * @default 128
             */
            min_overlap?: number;
            /**
             * type
             * @default calculate_image_tiles_min_overlap
             * @constant
             */
            type: "calculate_image_tiles_min_overlap";
        };
        /** CalculateImageTilesOutput */
        CalculateImageTilesOutput: {
            /**
             * Tiles
             * @description The tiles coordinates that cover a particular image shape.
             */
            tiles: components["schemas"]["Tile"][];
            /**
             * type
             * @default calculate_image_tiles_output
             * @constant
             */
            type: "calculate_image_tiles_output";
        };
        /**
         * CancelAllExceptCurrentResult
         * @description Result of canceling all except current
         */
        CancelAllExceptCurrentResult: {
            /**
             * Canceled
             * @description Number of queue items canceled
             */
            canceled: number;
        };
        /**
         * CancelByBatchIDsResult
         * @description Result of canceling by list of batch ids
         */
        CancelByBatchIDsResult: {
            /**
             * Canceled
             * @description Number of queue items canceled
             */
            canceled: number;
        };
        /**
         * CancelByDestinationResult
         * @description Result of canceling by a destination
         */
        CancelByDestinationResult: {
            /**
             * Canceled
             * @description Number of queue items canceled
             */
            canceled: number;
        };
        /**
         * Canny Edge Detection
         * @description Geneartes an edge map using a cv2's Canny algorithm.
         */
        CannyEdgeDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Low Threshold
             * @description The low threshold of the Canny pixel gradient (0-255)
             * @default 100
             */
            low_threshold?: number;
            /**
             * High Threshold
             * @description The high threshold of the Canny pixel gradient (0-255)
             * @default 200
             */
            high_threshold?: number;
            /**
             * type
             * @default canny_edge_detection
             * @constant
             */
            type: "canny_edge_detection";
        };
        /**
         * Canvas Paste Back
         * @description Combines two images by using the mask provided. Intended for use on the Unified Canvas.
         */
        CanvasPasteBackInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The source image
             * @default null
             */
            source_image?: components["schemas"]["ImageField"];
            /**
             * @description The target image
             * @default null
             */
            target_image?: components["schemas"]["ImageField"];
            /**
             * @description The mask to use when pasting
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * Mask Blur
             * @description The amount to blur the mask by
             * @default 0
             */
            mask_blur?: number;
            /**
             * type
             * @default canvas_paste_back
             * @constant
             */
            type: "canvas_paste_back";
        };
        /**
         * Canvas V2 Mask and Crop
         * @description Handles Canvas V2 image output masking and cropping
         */
        CanvasV2MaskAndCropInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The source image onto which the masked generated image is pasted. If omitted, the masked generated image is returned with transparency.
             * @default null
             */
            source_image?: components["schemas"]["ImageField"] | null;
            /**
             * @description The image to apply the mask to
             * @default null
             */
            generated_image?: components["schemas"]["ImageField"];
            /**
             * @description The mask to apply
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * Mask Blur
             * @description The amount to blur the mask by
             * @default 0
             */
            mask_blur?: number;
            /**
             * type
             * @default canvas_v2_mask_and_crop
             * @constant
             */
            type: "canvas_v2_mask_and_crop";
        };
        /**
         * Center Pad or Crop Image
         * @description Pad or crop an image's sides from the center by specified pixels. Positive values are outside of the image.
         */
        CenterPadCropInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to crop
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Left
             * @description Number of pixels to pad/crop from the left (negative values crop inwards, positive values pad outwards)
             * @default 0
             */
            left?: number;
            /**
             * Right
             * @description Number of pixels to pad/crop from the right (negative values crop inwards, positive values pad outwards)
             * @default 0
             */
            right?: number;
            /**
             * Top
             * @description Number of pixels to pad/crop from the top (negative values crop inwards, positive values pad outwards)
             * @default 0
             */
            top?: number;
            /**
             * Bottom
             * @description Number of pixels to pad/crop from the bottom (negative values crop inwards, positive values pad outwards)
             * @default 0
             */
            bottom?: number;
            /**
             * type
             * @default img_pad_crop
             * @constant
             */
            type: "img_pad_crop";
        };
        /**
         * Classification
         * @description The classification of an Invocation.
         *     - `Stable`: The invocation, including its inputs/outputs and internal logic, is stable. You may build workflows with it, having confidence that they will not break because of a change in this invocation.
         *     - `Beta`: The invocation is not yet stable, but is planned to be stable in the future. Workflows built around this invocation may break, but we are committed to supporting this invocation long-term.
         *     - `Prototype`: The invocation is not yet stable and may be removed from the application at any time. Workflows built around this invocation may break, and we are *not* committed to supporting this invocation.
         *     - `Deprecated`: The invocation is deprecated and may be removed in a future version.
         *     - `Internal`: The invocation is not intended for use by end-users. It may be changed or removed at any time, but is exposed for users to play with.
         *     - `Special`: The invocation is a special case and does not fit into any of the other classifications.
         * @enum {string}
         */
        Classification: "stable" | "beta" | "prototype" | "deprecated" | "internal" | "special";
        /**
         * ClearResult
         * @description Result of clearing the session queue
         */
        ClearResult: {
            /**
             * Deleted
             * @description Number of queue items deleted
             */
            deleted: number;
        };
        /**
         * ClipVariantType
         * @description Variant type.
         * @enum {string}
         */
        ClipVariantType: "large" | "gigantic";
        /**
         * CogView4ConditioningField
         * @description A conditioning tensor primitive value
         */
        CogView4ConditioningField: {
            /**
             * Conditioning Name
             * @description The name of conditioning tensor
             */
            conditioning_name: string;
        };
        /**
         * CogView4ConditioningOutput
         * @description Base class for nodes that output a CogView text conditioning tensor.
         */
        CogView4ConditioningOutput: {
            /** @description Conditioning tensor */
            conditioning: components["schemas"]["CogView4ConditioningField"];
            /**
             * type
             * @default cogview4_conditioning_output
             * @constant
             */
            type: "cogview4_conditioning_output";
        };
        /**
         * Denoise - CogView4
         * @description Run the denoising process with a CogView4 model.
         */
        CogView4DenoiseInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"] | null;
            /**
             * @description A mask of the region to apply the denoising process to. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved.
             * @default null
             */
            denoise_mask?: components["schemas"]["DenoiseMaskField"] | null;
            /**
             * Denoising Start
             * @description When to start denoising, expressed a percentage of total steps
             * @default 0
             */
            denoising_start?: number;
            /**
             * Denoising End
             * @description When to stop denoising, expressed a percentage of total steps
             * @default 1
             */
            denoising_end?: number;
            /**
             * Transformer
             * @description CogView4 model (Transformer) to load
             * @default null
             */
            transformer?: components["schemas"]["TransformerField"];
            /**
             * @description Positive conditioning tensor
             * @default null
             */
            positive_conditioning?: components["schemas"]["CogView4ConditioningField"];
            /**
             * @description Negative conditioning tensor
             * @default null
             */
            negative_conditioning?: components["schemas"]["CogView4ConditioningField"];
            /**
             * CFG Scale
             * @description Classifier-Free Guidance scale
             * @default 3.5
             */
            cfg_scale?: number | number[];
            /**
             * Width
             * @description Width of the generated image.
             * @default 1024
             */
            width?: number;
            /**
             * Height
             * @description Height of the generated image.
             * @default 1024
             */
            height?: number;
            /**
             * Steps
             * @description Number of steps to run
             * @default 25
             */
            steps?: number;
            /**
             * Seed
             * @description Randomness seed for reproducibility.
             * @default 0
             */
            seed?: number;
            /**
             * type
             * @default cogview4_denoise
             * @constant
             */
            type: "cogview4_denoise";
        };
        /**
         * Image to Latents - CogView4
         * @description Generates latents from an image.
         */
        CogView4ImageToLatentsInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to encode.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * type
             * @default cogview4_i2l
             * @constant
             */
            type: "cogview4_i2l";
        };
        /**
         * Latents to Image - CogView4
         * @description Generates an image from latents.
         */
        CogView4LatentsToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * type
             * @default cogview4_l2i
             * @constant
             */
            type: "cogview4_l2i";
        };
        /**
         * Main Model - CogView4
         * @description Loads a CogView4 base model, outputting its submodels.
         */
        CogView4ModelLoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /** @description CogView4 model (Transformer) to load */
            model: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default cogview4_model_loader
             * @constant
             */
            type: "cogview4_model_loader";
        };
        /**
         * CogView4ModelLoaderOutput
         * @description CogView4 base model loader output.
         */
        CogView4ModelLoaderOutput: {
            /**
             * Transformer
             * @description Transformer
             */
            transformer: components["schemas"]["TransformerField"];
            /**
             * GLM Encoder
             * @description GLM (THUDM) tokenizer and text encoder
             */
            glm_encoder: components["schemas"]["GlmEncoderField"];
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * type
             * @default cogview4_model_loader_output
             * @constant
             */
            type: "cogview4_model_loader_output";
        };
        /**
         * Prompt - CogView4
         * @description Encodes and preps a prompt for a cogview4 image.
         */
        CogView4TextEncoderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Prompt
             * @description Text prompt to encode.
             * @default null
             */
            prompt?: string;
            /**
             * GLM Encoder
             * @description GLM (THUDM) tokenizer and text encoder
             * @default null
             */
            glm_encoder?: components["schemas"]["GlmEncoderField"];
            /**
             * type
             * @default cogview4_text_encoder
             * @constant
             */
            type: "cogview4_text_encoder";
        };
        /**
         * CollectInvocation
         * @description Collects values into a collection
         */
        CollectInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection Item
             * @description The item to collect (all inputs must be of the same type)
             * @default null
             */
            item?: unknown | null;
            /**
             * Collection
             * @description The collection, will be provided on execution
             * @default []
             */
            collection?: unknown[];
            /**
             * type
             * @default collect
             * @constant
             */
            type: "collect";
        };
        /** CollectInvocationOutput */
        CollectInvocationOutput: {
            /**
             * Collection
             * @description The collection of input items
             */
            collection: unknown[];
            /**
             * type
             * @default collect_output
             * @constant
             */
            type: "collect_output";
        };
        /**
         * ColorCollectionOutput
         * @description Base class for nodes that output a collection of colors
         */
        ColorCollectionOutput: {
            /**
             * Collection
             * @description The output colors
             */
            collection: components["schemas"]["ColorField"][];
            /**
             * type
             * @default color_collection_output
             * @constant
             */
            type: "color_collection_output";
        };
        /**
         * Color Correct
         * @description Shifts the colors of a target image to match the reference image, optionally
         *     using a mask to only color-correct certain regions of the target image.
         */
        ColorCorrectInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to color-correct
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description Reference image for color-correction
             * @default null
             */
            reference?: components["schemas"]["ImageField"];
            /**
             * @description Mask to use when applying color-correction
             * @default null
             */
            mask?: components["schemas"]["ImageField"] | null;
            /**
             * Mask Blur Radius
             * @description Mask blur radius
             * @default 8
             */
            mask_blur_radius?: number;
            /**
             * type
             * @default color_correct
             * @constant
             */
            type: "color_correct";
        };
        /**
         * ColorField
         * @description A color primitive field
         */
        ColorField: {
            /**
             * R
             * @description The red component
             */
            r: number;
            /**
             * G
             * @description The green component
             */
            g: number;
            /**
             * B
             * @description The blue component
             */
            b: number;
            /**
             * A
             * @description The alpha component
             */
            a: number;
        };
        /**
         * Color Primitive
         * @description A color primitive value
         */
        ColorInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The color value
             * @default {
             *       "r": 0,
             *       "g": 0,
             *       "b": 0,
             *       "a": 255
             *     }
             */
            color?: components["schemas"]["ColorField"];
            /**
             * type
             * @default color
             * @constant
             */
            type: "color";
        };
        /**
         * Color Map
         * @description Generates a color map from the provided image.
         */
        ColorMapInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Tile Size
             * @description Tile size
             * @default 64
             */
            tile_size?: number;
            /**
             * type
             * @default color_map
             * @constant
             */
            type: "color_map";
        };
        /**
         * ColorOutput
         * @description Base class for nodes that output a single color
         */
        ColorOutput: {
            /** @description The output color */
            color: components["schemas"]["ColorField"];
            /**
             * type
             * @default color_output
             * @constant
             */
            type: "color_output";
        };
        /**
         * Prompt - SD1.5
         * @description Parse prompt using compel package to conditioning.
         */
        CompelInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Prompt
             * @description Prompt to be parsed by Compel to create a conditioning tensor
             * @default
             */
            prompt?: string;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"];
            /**
             * @description A mask defining the region that this conditioning prompt applies to.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
            /**
             * type
             * @default compel
             * @constant
             */
            type: "compel";
        };
        /**
         * Conditioning Collection Primitive
         * @description A collection of conditioning tensor primitive values
         */
        ConditioningCollectionInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The collection of conditioning tensors
             * @default []
             */
            collection?: components["schemas"]["ConditioningField"][];
            /**
             * type
             * @default conditioning_collection
             * @constant
             */
            type: "conditioning_collection";
        };
        /**
         * ConditioningCollectionOutput
         * @description Base class for nodes that output a collection of conditioning tensors
         */
        ConditioningCollectionOutput: {
            /**
             * Collection
             * @description The output conditioning tensors
             */
            collection: components["schemas"]["ConditioningField"][];
            /**
             * type
             * @default conditioning_collection_output
             * @constant
             */
            type: "conditioning_collection_output";
        };
        /**
         * ConditioningField
         * @description A conditioning tensor primitive value
         */
        ConditioningField: {
            /**
             * Conditioning Name
             * @description The name of conditioning tensor
             */
            conditioning_name: string;
            /**
             * @description The mask associated with this conditioning tensor. Excluded regions should be set to False, included regions should be set to True.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
        };
        /**
         * Conditioning Primitive
         * @description A conditioning tensor primitive value
         */
        ConditioningInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Conditioning tensor
             * @default null
             */
            conditioning?: components["schemas"]["ConditioningField"];
            /**
             * type
             * @default conditioning
             * @constant
             */
            type: "conditioning";
        };
        /**
         * ConditioningOutput
         * @description Base class for nodes that output a single conditioning tensor
         */
        ConditioningOutput: {
            /** @description Conditioning tensor */
            conditioning: components["schemas"]["ConditioningField"];
            /**
             * type
             * @default conditioning_output
             * @constant
             */
            type: "conditioning_output";
        };
        /**
         * Content Shuffle
         * @description Shuffles the image, similar to a 'liquify' filter.
         */
        ContentShuffleInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Scale Factor
             * @description The scale factor used for the shuffle
             * @default 256
             */
            scale_factor?: number;
            /**
             * type
             * @default content_shuffle
             * @constant
             */
            type: "content_shuffle";
        };
        /** ControlAdapterDefaultSettings */
        ControlAdapterDefaultSettings: {
            /** Preprocessor */
            preprocessor: string | null;
        };
        /** ControlField */
        ControlField: {
            /** @description The control image */
            image: components["schemas"]["ImageField"];
            /** @description The ControlNet model to use */
            control_model: components["schemas"]["ModelIdentifierField"];
            /**
             * Control Weight
             * @description The weight given to the ControlNet
             * @default 1
             */
            control_weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the ControlNet is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the ControlNet is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Control Mode
             * @description The control mode to use
             * @default balanced
             * @enum {string}
             */
            control_mode?: "balanced" | "more_prompt" | "more_control" | "unbalanced";
            /**
             * Resize Mode
             * @description The resize mode to use
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
        };
        /**
         * ControlLoRADiffusersConfig
         * @description Model config for Control LoRA models.
         */
        ControlLoRADiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default control_lora
             * @constant
             */
            type: "control_lora";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["ControlAdapterDefaultSettings"] | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
        };
        /** ControlLoRAField */
        ControlLoRAField: {
            /** @description Info to load lora model */
            lora: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description Weight to apply to lora model
             */
            weight: number;
            /** @description Image to use in structural conditioning */
            img: components["schemas"]["ImageField"];
        };
        /**
         * ControlLoRALyCORISConfig
         * @description Model config for Control LoRA models.
         */
        ControlLoRALyCORISConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default control_lora
             * @constant
             */
            type: "control_lora";
            /**
             * Format
             * @default lycoris
             * @constant
             */
            format: "lycoris";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["ControlAdapterDefaultSettings"] | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
        };
        /**
         * ControlNetCheckpointConfig
         * @description Model config for ControlNet models (diffusers version).
         */
        ControlNetCheckpointConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default controlnet
             * @constant
             */
            type: "controlnet";
            /**
             * Format
             * @description Format of the provided checkpoint model
             * @default checkpoint
             * @enum {string}
             */
            format: "checkpoint" | "bnb_quantized_nf4b" | "gguf_quantized";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["ControlAdapterDefaultSettings"] | null;
            /**
             * Config Path
             * @description path to the checkpoint model config file
             */
            config_path: string;
            /**
             * Converted At
             * @description When this model was last converted to diffusers
             */
            converted_at?: number | null;
        };
        /**
         * ControlNetDiffusersConfig
         * @description Model config for ControlNet models (diffusers version).
         */
        ControlNetDiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default controlnet
             * @constant
             */
            type: "controlnet";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["ControlAdapterDefaultSettings"] | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
        };
        /**
         * ControlNet - SD1.5, SDXL
         * @description Collects ControlNet info to pass to other nodes
         */
        ControlNetInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The control image
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description ControlNet model to load
             * @default null
             */
            control_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Control Weight
             * @description The weight given to the ControlNet
             * @default 1
             */
            control_weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the ControlNet is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the ControlNet is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Control Mode
             * @description The control mode used
             * @default balanced
             * @enum {string}
             */
            control_mode?: "balanced" | "more_prompt" | "more_control" | "unbalanced";
            /**
             * Resize Mode
             * @description The resize mode used
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
            /**
             * type
             * @default controlnet
             * @constant
             */
            type: "controlnet";
        };
        /** ControlNetMetadataField */
        ControlNetMetadataField: {
            /** @description The control image */
            image: components["schemas"]["ImageField"];
            /**
             * @description The control image, after processing.
             * @default null
             */
            processed_image?: components["schemas"]["ImageField"] | null;
            /** @description The ControlNet model to use */
            control_model: components["schemas"]["ModelIdentifierField"];
            /**
             * Control Weight
             * @description The weight given to the ControlNet
             * @default 1
             */
            control_weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the ControlNet is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the ControlNet is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Control Mode
             * @description The control mode to use
             * @default balanced
             * @enum {string}
             */
            control_mode?: "balanced" | "more_prompt" | "more_control" | "unbalanced";
            /**
             * Resize Mode
             * @description The resize mode to use
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
        };
        /**
         * ControlOutput
         * @description node output for ControlNet info
         */
        ControlOutput: {
            /** @description ControlNet(s) to apply */
            control: components["schemas"]["ControlField"];
            /**
             * type
             * @default control_output
             * @constant
             */
            type: "control_output";
        };
        /**
         * Core Metadata
         * @description Used internally by Invoke to collect metadata for generations.
         */
        CoreMetadataInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Generation Mode
             * @description The generation mode that output this image
             * @default null
             */
            generation_mode?: ("txt2img" | "img2img" | "inpaint" | "outpaint" | "sdxl_txt2img" | "sdxl_img2img" | "sdxl_inpaint" | "sdxl_outpaint" | "flux_txt2img" | "flux_img2img" | "flux_inpaint" | "flux_outpaint" | "sd3_txt2img" | "sd3_img2img" | "sd3_inpaint" | "sd3_outpaint" | "cogview4_txt2img" | "cogview4_img2img" | "cogview4_inpaint" | "cogview4_outpaint") | null;
            /**
             * Positive Prompt
             * @description The positive prompt parameter
             * @default null
             */
            positive_prompt?: string | null;
            /**
             * Negative Prompt
             * @description The negative prompt parameter
             * @default null
             */
            negative_prompt?: string | null;
            /**
             * Width
             * @description The width parameter
             * @default null
             */
            width?: number | null;
            /**
             * Height
             * @description The height parameter
             * @default null
             */
            height?: number | null;
            /**
             * Seed
             * @description The seed used for noise generation
             * @default null
             */
            seed?: number | null;
            /**
             * Rand Device
             * @description The device used for random number generation
             * @default null
             */
            rand_device?: string | null;
            /**
             * Cfg Scale
             * @description The classifier-free guidance scale parameter
             * @default null
             */
            cfg_scale?: number | null;
            /**
             * Cfg Rescale Multiplier
             * @description Rescale multiplier for CFG guidance, used for models trained with zero-terminal SNR
             * @default null
             */
            cfg_rescale_multiplier?: number | null;
            /**
             * Steps
             * @description The number of steps used for inference
             * @default null
             */
            steps?: number | null;
            /**
             * Scheduler
             * @description The scheduler used for inference
             * @default null
             */
            scheduler?: string | null;
            /**
             * Seamless X
             * @description Whether seamless tiling was used on the X axis
             * @default null
             */
            seamless_x?: boolean | null;
            /**
             * Seamless Y
             * @description Whether seamless tiling was used on the Y axis
             * @default null
             */
            seamless_y?: boolean | null;
            /**
             * Clip Skip
             * @description The number of skipped CLIP layers
             * @default null
             */
            clip_skip?: number | null;
            /**
             * @description The main model used for inference
             * @default null
             */
            model?: components["schemas"]["ModelIdentifierField"] | null;
            /**
             * Controlnets
             * @description The ControlNets used for inference
             * @default null
             */
            controlnets?: components["schemas"]["ControlNetMetadataField"][] | null;
            /**
             * Ipadapters
             * @description The IP Adapters used for inference
             * @default null
             */
            ipAdapters?: components["schemas"]["IPAdapterMetadataField"][] | null;
            /**
             * T2Iadapters
             * @description The IP Adapters used for inference
             * @default null
             */
            t2iAdapters?: components["schemas"]["T2IAdapterMetadataField"][] | null;
            /**
             * Loras
             * @description The LoRAs used for inference
             * @default null
             */
            loras?: components["schemas"]["LoRAMetadataField"][] | null;
            /**
             * Strength
             * @description The strength used for latents-to-latents
             * @default null
             */
            strength?: number | null;
            /**
             * Init Image
             * @description The name of the initial image
             * @default null
             */
            init_image?: string | null;
            /**
             * @description The VAE used for decoding, if the main model's default was not used
             * @default null
             */
            vae?: components["schemas"]["ModelIdentifierField"] | null;
            /**
             * Hrf Enabled
             * @description Whether or not high resolution fix was enabled.
             * @default null
             */
            hrf_enabled?: boolean | null;
            /**
             * Hrf Method
             * @description The high resolution fix upscale method.
             * @default null
             */
            hrf_method?: string | null;
            /**
             * Hrf Strength
             * @description The high resolution fix img2img strength used in the upscale pass.
             * @default null
             */
            hrf_strength?: number | null;
            /**
             * Positive Style Prompt
             * @description The positive style prompt parameter
             * @default null
             */
            positive_style_prompt?: string | null;
            /**
             * Negative Style Prompt
             * @description The negative style prompt parameter
             * @default null
             */
            negative_style_prompt?: string | null;
            /**
             * @description The SDXL Refiner model used
             * @default null
             */
            refiner_model?: components["schemas"]["ModelIdentifierField"] | null;
            /**
             * Refiner Cfg Scale
             * @description The classifier-free guidance scale parameter used for the refiner
             * @default null
             */
            refiner_cfg_scale?: number | null;
            /**
             * Refiner Steps
             * @description The number of steps used for the refiner
             * @default null
             */
            refiner_steps?: number | null;
            /**
             * Refiner Scheduler
             * @description The scheduler used for the refiner
             * @default null
             */
            refiner_scheduler?: string | null;
            /**
             * Refiner Positive Aesthetic Score
             * @description The aesthetic score used for the refiner
             * @default null
             */
            refiner_positive_aesthetic_score?: number | null;
            /**
             * Refiner Negative Aesthetic Score
             * @description The aesthetic score used for the refiner
             * @default null
             */
            refiner_negative_aesthetic_score?: number | null;
            /**
             * Refiner Start
             * @description The start value used for refiner denoising
             * @default null
             */
            refiner_start?: number | null;
            /**
             * type
             * @default core_metadata
             * @constant
             */
            type: "core_metadata";
        } & {
            [key: string]: unknown;
        };
        /**
         * Create Denoise Mask
         * @description Creates mask for denoising model run.
         */
        CreateDenoiseMaskInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * @description Image which will be masked
             * @default null
             */
            image?: components["schemas"]["ImageField"] | null;
            /**
             * @description The mask to use when pasting
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * Tiled
             * @description Processing using overlapping tiles (reduce memory consumption)
             * @default false
             */
            tiled?: boolean;
            /**
             * Fp32
             * @description Whether or not to use full float32 precision
             * @default false
             */
            fp32?: boolean;
            /**
             * type
             * @default create_denoise_mask
             * @constant
             */
            type: "create_denoise_mask";
        };
        /**
         * Create Gradient Mask
         * @description Creates mask for denoising model run.
         */
        CreateGradientMaskInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Image which will be masked
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * Edge Radius
             * @description How far to blur/expand the edges of the mask
             * @default 16
             */
            edge_radius?: number;
            /**
             * Coherence Mode
             * @default Gaussian Blur
             * @enum {string}
             */
            coherence_mode?: "Gaussian Blur" | "Box Blur" | "Staged";
            /**
             * Minimum Denoise
             * @description Minimum denoise level for the coherence region
             * @default 0
             */
            minimum_denoise?: number;
            /**
             * [OPTIONAL] Image
             * @description OPTIONAL: Only connect for specialized Inpainting models, masked_latents will be generated from the image with the VAE
             * @default null
             */
            image?: components["schemas"]["ImageField"] | null;
            /**
             * [OPTIONAL] UNet
             * @description OPTIONAL: If the Unet is a specialized Inpainting model, masked_latents will be generated from the image with the VAE
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * [OPTIONAL] VAE
             * @description OPTIONAL: Only connect for specialized Inpainting models, masked_latents will be generated from the image with the VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"] | null;
            /**
             * Tiled
             * @description Processing using overlapping tiles (reduce memory consumption)
             * @default false
             */
            tiled?: boolean;
            /**
             * Fp32
             * @description Whether or not to use full float32 precision
             * @default false
             */
            fp32?: boolean;
            /**
             * type
             * @default create_gradient_mask
             * @constant
             */
            type: "create_gradient_mask";
        };
        /**
         * Crop Image to Bounding Box
         * @description Crop an image to the given bounding box. If the bounding box is omitted, the image is cropped to the non-transparent pixels.
         */
        CropImageToBoundingBoxInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to crop
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The bounding box to crop the image to
             * @default null
             */
            bounding_box?: components["schemas"]["BoundingBoxField"] | null;
            /**
             * type
             * @default crop_image_to_bounding_box
             * @constant
             */
            type: "crop_image_to_bounding_box";
        };
        /**
         * Crop Latents
         * @description Crops a latent-space tensor to a box specified in image-space. The box dimensions and coordinates must be
         *     divisible by the latent scale factor of 8.
         */
        CropLatentsCoreInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * X
             * @description The left x coordinate (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.
             * @default null
             */
            x?: number;
            /**
             * Y
             * @description The top y coordinate (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.
             * @default null
             */
            y?: number;
            /**
             * Width
             * @description The width (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.
             * @default null
             */
            width?: number;
            /**
             * Height
             * @description The height (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.
             * @default null
             */
            height?: number;
            /**
             * type
             * @default crop_latents
             * @constant
             */
            type: "crop_latents";
        };
        /** CursorPaginatedResults[SessionQueueItemDTO] */
        CursorPaginatedResults_SessionQueueItemDTO_: {
            /**
             * Limit
             * @description Limit of items to get
             */
            limit: number;
            /**
             * Has More
             * @description Whether there are more items available
             */
            has_more: boolean;
            /**
             * Items
             * @description Items
             */
            items: components["schemas"]["SessionQueueItemDTO"][];
        };
        /**
         * OpenCV Inpaint
         * @description Simple inpaint using opencv.
         */
        CvInpaintInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to inpaint
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The mask to use when inpainting
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * type
             * @default cv_inpaint
             * @constant
             */
            type: "cv_inpaint";
        };
        /**
         * DW Openpose Detection
         * @description Generates an openpose pose from an image using DWPose
         */
        DWOpenposeDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Draw Body
             * @default true
             */
            draw_body?: boolean;
            /**
             * Draw Face
             * @default false
             */
            draw_face?: boolean;
            /**
             * Draw Hands
             * @default false
             */
            draw_hands?: boolean;
            /**
             * type
             * @default dw_openpose_detection
             * @constant
             */
            type: "dw_openpose_detection";
        };
        /** DeleteBoardResult */
        DeleteBoardResult: {
            /**
             * Board Id
             * @description The id of the board that was deleted.
             */
            board_id: string;
            /**
             * Deleted Board Images
             * @description The image names of the board-images relationships that were deleted.
             */
            deleted_board_images: string[];
            /**
             * Deleted Images
             * @description The names of the images that were deleted.
             */
            deleted_images: string[];
        };
        /** DeleteImagesFromListResult */
        DeleteImagesFromListResult: {
            /** Deleted Images */
            deleted_images: string[];
        };
        /**
         * Denoise - SD1.5, SDXL
         * @description Denoises noisy latents to decodable images
         */
        DenoiseLatentsInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Positive Conditioning
             * @description Positive conditioning tensor
             * @default null
             */
            positive_conditioning?: components["schemas"]["ConditioningField"] | components["schemas"]["ConditioningField"][];
            /**
             * Negative Conditioning
             * @description Negative conditioning tensor
             * @default null
             */
            negative_conditioning?: components["schemas"]["ConditioningField"] | components["schemas"]["ConditioningField"][];
            /**
             * @description Noise tensor
             * @default null
             */
            noise?: components["schemas"]["LatentsField"] | null;
            /**
             * Steps
             * @description Number of steps to run
             * @default 10
             */
            steps?: number;
            /**
             * CFG Scale
             * @description Classifier-Free Guidance scale
             * @default 7.5
             */
            cfg_scale?: number | number[];
            /**
             * Denoising Start
             * @description When to start denoising, expressed a percentage of total steps
             * @default 0
             */
            denoising_start?: number;
            /**
             * Denoising End
             * @description When to stop denoising, expressed a percentage of total steps
             * @default 1
             */
            denoising_end?: number;
            /**
             * Scheduler
             * @description Scheduler to use during inference
             * @default euler
             * @enum {string}
             */
            scheduler?: "ddim" | "ddpm" | "deis" | "deis_k" | "lms" | "lms_k" | "pndm" | "heun" | "heun_k" | "euler" | "euler_k" | "euler_a" | "kdpm_2" | "kdpm_2_k" | "kdpm_2_a" | "kdpm_2_a_k" | "dpmpp_2s" | "dpmpp_2s_k" | "dpmpp_2m" | "dpmpp_2m_k" | "dpmpp_2m_sde" | "dpmpp_2m_sde_k" | "dpmpp_3m" | "dpmpp_3m_k" | "dpmpp_sde" | "dpmpp_sde_k" | "unipc" | "unipc_k" | "lcm" | "tcd";
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"];
            /**
             * Control
             * @default null
             */
            control?: components["schemas"]["ControlField"] | components["schemas"]["ControlField"][] | null;
            /**
             * IP-Adapter
             * @description IP-Adapter to apply
             * @default null
             */
            ip_adapter?: components["schemas"]["IPAdapterField"] | components["schemas"]["IPAdapterField"][] | null;
            /**
             * T2I-Adapter
             * @description T2I-Adapter(s) to apply
             * @default null
             */
            t2i_adapter?: components["schemas"]["T2IAdapterField"] | components["schemas"]["T2IAdapterField"][] | null;
            /**
             * CFG Rescale Multiplier
             * @description Rescale multiplier for CFG guidance, used for models trained with zero-terminal SNR
             * @default 0
             */
            cfg_rescale_multiplier?: number;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"] | null;
            /**
             * @description A mask of the region to apply the denoising process to. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved.
             * @default null
             */
            denoise_mask?: components["schemas"]["DenoiseMaskField"] | null;
            /**
             * type
             * @default denoise_latents
             * @constant
             */
            type: "denoise_latents";
        };
        /** Denoise - SD1.5, SDXL + Metadata */
        DenoiseLatentsMetaInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Positive Conditioning
             * @description Positive conditioning tensor
             * @default null
             */
            positive_conditioning?: components["schemas"]["ConditioningField"] | components["schemas"]["ConditioningField"][];
            /**
             * Negative Conditioning
             * @description Negative conditioning tensor
             * @default null
             */
            negative_conditioning?: components["schemas"]["ConditioningField"] | components["schemas"]["ConditioningField"][];
            /**
             * @description Noise tensor
             * @default null
             */
            noise?: components["schemas"]["LatentsField"] | null;
            /**
             * Steps
             * @description Number of steps to run
             * @default 10
             */
            steps?: number;
            /**
             * CFG Scale
             * @description Classifier-Free Guidance scale
             * @default 7.5
             */
            cfg_scale?: number | number[];
            /**
             * Denoising Start
             * @description When to start denoising, expressed a percentage of total steps
             * @default 0
             */
            denoising_start?: number;
            /**
             * Denoising End
             * @description When to stop denoising, expressed a percentage of total steps
             * @default 1
             */
            denoising_end?: number;
            /**
             * Scheduler
             * @description Scheduler to use during inference
             * @default euler
             * @enum {string}
             */
            scheduler?: "ddim" | "ddpm" | "deis" | "deis_k" | "lms" | "lms_k" | "pndm" | "heun" | "heun_k" | "euler" | "euler_k" | "euler_a" | "kdpm_2" | "kdpm_2_k" | "kdpm_2_a" | "kdpm_2_a_k" | "dpmpp_2s" | "dpmpp_2s_k" | "dpmpp_2m" | "dpmpp_2m_k" | "dpmpp_2m_sde" | "dpmpp_2m_sde_k" | "dpmpp_3m" | "dpmpp_3m_k" | "dpmpp_sde" | "dpmpp_sde_k" | "unipc" | "unipc_k" | "lcm" | "tcd";
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"];
            /**
             * Control
             * @default null
             */
            control?: components["schemas"]["ControlField"] | components["schemas"]["ControlField"][] | null;
            /**
             * IP-Adapter
             * @description IP-Adapter to apply
             * @default null
             */
            ip_adapter?: components["schemas"]["IPAdapterField"] | components["schemas"]["IPAdapterField"][] | null;
            /**
             * T2I-Adapter
             * @description T2I-Adapter(s) to apply
             * @default null
             */
            t2i_adapter?: components["schemas"]["T2IAdapterField"] | components["schemas"]["T2IAdapterField"][] | null;
            /**
             * CFG Rescale Multiplier
             * @description Rescale multiplier for CFG guidance, used for models trained with zero-terminal SNR
             * @default 0
             */
            cfg_rescale_multiplier?: number;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"] | null;
            /**
             * @description A mask of the region to apply the denoising process to. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved.
             * @default null
             */
            denoise_mask?: components["schemas"]["DenoiseMaskField"] | null;
            /**
             * type
             * @default denoise_latents_meta
             * @constant
             */
            type: "denoise_latents_meta";
        };
        /**
         * DenoiseMaskField
         * @description An inpaint mask field
         */
        DenoiseMaskField: {
            /**
             * Mask Name
             * @description The name of the mask image
             */
            mask_name: string;
            /**
             * Masked Latents Name
             * @description The name of the masked image latents
             * @default null
             */
            masked_latents_name?: string | null;
            /**
             * Gradient
             * @description Used for gradient inpainting
             * @default false
             */
            gradient?: boolean;
        };
        /**
         * DenoiseMaskOutput
         * @description Base class for nodes that output a single image
         */
        DenoiseMaskOutput: {
            /** @description Mask for denoise model run */
            denoise_mask: components["schemas"]["DenoiseMaskField"];
            /**
             * type
             * @default denoise_mask_output
             * @constant
             */
            type: "denoise_mask_output";
        };
        /**
         * Depth Anything Depth Estimation
         * @description Generates a depth map using a Depth Anything model.
         */
        DepthAnythingDepthEstimationInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Model Size
             * @description The size of the depth model to use
             * @default small_v2
             * @enum {string}
             */
            model_size?: "large" | "base" | "small" | "small_v2";
            /**
             * type
             * @default depth_anything_depth_estimation
             * @constant
             */
            type: "depth_anything_depth_estimation";
        };
        /**
         * Divide Integers
         * @description Divides two numbers
         */
        DivideInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * A
             * @description The first number
             * @default 0
             */
            a?: number;
            /**
             * B
             * @description The second number
             * @default 0
             */
            b?: number;
            /**
             * type
             * @default div
             * @constant
             */
            type: "div";
        };
        /**
         * DownloadCancelledEvent
         * @description Event model for download_cancelled
         */
        DownloadCancelledEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Source
             * @description The source of the download
             */
            source: string;
        };
        /**
         * DownloadCompleteEvent
         * @description Event model for download_complete
         */
        DownloadCompleteEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Source
             * @description The source of the download
             */
            source: string;
            /**
             * Download Path
             * @description The local path where the download is saved
             */
            download_path: string;
            /**
             * Total Bytes
             * @description The total number of bytes downloaded
             */
            total_bytes: number;
        };
        /**
         * DownloadErrorEvent
         * @description Event model for download_error
         */
        DownloadErrorEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Source
             * @description The source of the download
             */
            source: string;
            /**
             * Error Type
             * @description The type of error
             */
            error_type: string;
            /**
             * Error
             * @description The error message
             */
            error: string;
        };
        /**
         * DownloadJob
         * @description Class to monitor and control a model download request.
         */
        DownloadJob: {
            /**
             * Id
             * @description Numeric ID of this job
             * @default -1
             */
            id?: number;
            /**
             * Dest
             * Format: path
             * @description Initial destination of downloaded model on local disk; a directory or file path
             */
            dest: string;
            /**
             * Download Path
             * @description Final location of downloaded file or directory
             */
            download_path?: string | null;
            /**
             * @description Status of the download
             * @default waiting
             */
            status?: components["schemas"]["DownloadJobStatus"];
            /**
             * Bytes
             * @description Bytes downloaded so far
             * @default 0
             */
            bytes?: number;
            /**
             * Total Bytes
             * @description Total file size (bytes)
             * @default 0
             */
            total_bytes?: number;
            /**
             * Error Type
             * @description Name of exception that caused an error
             */
            error_type?: string | null;
            /**
             * Error
             * @description Traceback of the exception that caused an error
             */
            error?: string | null;
            /**
             * Source
             * Format: uri
             * @description Where to download from. Specific types specified in child classes.
             */
            source: string;
            /**
             * Access Token
             * @description authorization token for protected resources
             */
            access_token?: string | null;
            /**
             * Priority
             * @description Queue priority; lower values are higher priority
             * @default 10
             */
            priority?: number;
            /**
             * Job Started
             * @description Timestamp for when the download job started
             */
            job_started?: string | null;
            /**
             * Job Ended
             * @description Timestamp for when the download job ende1d (completed or errored)
             */
            job_ended?: string | null;
            /**
             * Content Type
             * @description Content type of downloaded file
             */
            content_type?: string | null;
        };
        /**
         * DownloadJobStatus
         * @description State of a download job.
         * @enum {string}
         */
        DownloadJobStatus: "waiting" | "running" | "completed" | "cancelled" | "error";
        /**
         * DownloadProgressEvent
         * @description Event model for download_progress
         */
        DownloadProgressEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Source
             * @description The source of the download
             */
            source: string;
            /**
             * Download Path
             * @description The local path where the download is saved
             */
            download_path: string;
            /**
             * Current Bytes
             * @description The number of bytes downloaded so far
             */
            current_bytes: number;
            /**
             * Total Bytes
             * @description The total number of bytes to be downloaded
             */
            total_bytes: number;
        };
        /**
         * DownloadStartedEvent
         * @description Event model for download_started
         */
        DownloadStartedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Source
             * @description The source of the download
             */
            source: string;
            /**
             * Download Path
             * @description The local path where the download is saved
             */
            download_path: string;
        };
        /**
         * Dynamic Prompt
         * @description Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator
         */
        DynamicPromptInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default false
             */
            use_cache?: boolean;
            /**
             * Prompt
             * @description The prompt to parse with dynamicprompts
             * @default null
             */
            prompt?: string;
            /**
             * Max Prompts
             * @description The number of prompts to generate
             * @default 1
             */
            max_prompts?: number;
            /**
             * Combinatorial
             * @description Whether to use the combinatorial generator
             * @default false
             */
            combinatorial?: boolean;
            /**
             * type
             * @default dynamic_prompt
             * @constant
             */
            type: "dynamic_prompt";
        };
        /** DynamicPromptsResponse */
        DynamicPromptsResponse: {
            /** Prompts */
            prompts: string[];
            /** Error */
            error?: string | null;
        };
        /**
         * Upscale (RealESRGAN)
         * @description Upscales an image using RealESRGAN.
         */
        ESRGANInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The input image
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Model Name
             * @description The Real-ESRGAN model to use
             * @default RealESRGAN_x4plus.pth
             * @enum {string}
             */
            model_name?: "RealESRGAN_x4plus.pth" | "RealESRGAN_x4plus_anime_6B.pth" | "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth" | "RealESRGAN_x2plus.pth";
            /**
             * Tile Size
             * @description Tile size for tiled ESRGAN upscaling (0=tiling disabled)
             * @default 400
             */
            tile_size?: number;
            /**
             * type
             * @default esrgan
             * @constant
             */
            type: "esrgan";
        };
        /** Edge */
        Edge: {
            /** @description The connection for the edge's from node and field */
            source: components["schemas"]["EdgeConnection"];
            /** @description The connection for the edge's to node and field */
            destination: components["schemas"]["EdgeConnection"];
        };
        /** EdgeConnection */
        EdgeConnection: {
            /**
             * Node Id
             * @description The id of the node for this edge connection
             */
            node_id: string;
            /**
             * Field
             * @description The field for this connection
             */
            field: string;
        };
        /** EnqueueBatchResult */
        EnqueueBatchResult: {
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Enqueued
             * @description The total number of queue items enqueued
             */
            enqueued: number;
            /**
             * Requested
             * @description The total number of queue items requested to be enqueued
             */
            requested: number;
            /** @description The batch that was enqueued */
            batch: components["schemas"]["Batch"];
            /**
             * Priority
             * @description The priority of the enqueued batch
             */
            priority: number;
        };
        /**
         * Expand Mask with Fade
         * @description Expands a mask with a fade effect. The mask uses black to indicate areas to keep from the generated image and white for areas to discard.
         *     The mask is thresholded to create a binary mask, and then a distance transform is applied to create a fade effect.
         *     The fade size is specified in pixels, and the mask is expanded by that amount. The result is a mask with a smooth transition from black to white.
         *     If the fade size is 0, the mask is returned as-is.
         */
        ExpandMaskWithFadeInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The mask to expand
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * Threshold
             * @description The threshold for the binary mask (0-255)
             * @default 0
             */
            threshold?: number;
            /**
             * Fade Size Px
             * @description The size of the fade in pixels
             * @default 32
             */
            fade_size_px?: number;
            /**
             * type
             * @default expand_mask_with_fade
             * @constant
             */
            type: "expand_mask_with_fade";
        };
        /** ExposedField */
        ExposedField: {
            /** Nodeid */
            nodeId: string;
            /** Fieldname */
            fieldName: string;
        };
        /**
         * Apply LoRA Collection - FLUX
         * @description Applies a collection of LoRAs to a FLUX transformer.
         */
        FLUXLoRACollectionLoader: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * LoRAs
             * @description LoRA models and weights. May be a single LoRA or collection.
             * @default null
             */
            loras?: components["schemas"]["LoRAField"] | components["schemas"]["LoRAField"][] | null;
            /**
             * Transformer
             * @description Transformer
             * @default null
             */
            transformer?: components["schemas"]["TransformerField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * T5 Encoder
             * @description T5 tokenizer and text encoder
             * @default null
             */
            t5_encoder?: components["schemas"]["T5EncoderField"] | null;
            /**
             * type
             * @default flux_lora_collection_loader
             * @constant
             */
            type: "flux_lora_collection_loader";
        };
        /**
         * FaceIdentifier
         * @description Outputs an image with detected face IDs printed on each face. For use with other FaceTools.
         */
        FaceIdentifierInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Image to face detect
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Minimum Confidence
             * @description Minimum confidence for face detection (lower if detection is failing)
             * @default 0.5
             */
            minimum_confidence?: number;
            /**
             * Chunk
             * @description Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.
             * @default false
             */
            chunk?: boolean;
            /**
             * type
             * @default face_identifier
             * @constant
             */
            type: "face_identifier";
        };
        /**
         * FaceMask
         * @description Face mask creation using mediapipe face detection
         */
        FaceMaskInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Image to face detect
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Face Ids
             * @description Comma-separated list of face ids to mask eg '0,2,7'. Numbered from 0. Leave empty to mask all. Find face IDs with FaceIdentifier node.
             * @default
             */
            face_ids?: string;
            /**
             * Minimum Confidence
             * @description Minimum confidence for face detection (lower if detection is failing)
             * @default 0.5
             */
            minimum_confidence?: number;
            /**
             * X Offset
             * @description Offset for the X-axis of the face mask
             * @default 0
             */
            x_offset?: number;
            /**
             * Y Offset
             * @description Offset for the Y-axis of the face mask
             * @default 0
             */
            y_offset?: number;
            /**
             * Chunk
             * @description Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.
             * @default false
             */
            chunk?: boolean;
            /**
             * Invert Mask
             * @description Toggle to invert the mask
             * @default false
             */
            invert_mask?: boolean;
            /**
             * type
             * @default face_mask_detection
             * @constant
             */
            type: "face_mask_detection";
        };
        /**
         * FaceMaskOutput
         * @description Base class for FaceMask output
         */
        FaceMaskOutput: {
            /** @description The output image */
            image: components["schemas"]["ImageField"];
            /**
             * Width
             * @description The width of the image in pixels
             */
            width: number;
            /**
             * Height
             * @description The height of the image in pixels
             */
            height: number;
            /**
             * type
             * @default face_mask_output
             * @constant
             */
            type: "face_mask_output";
            /** @description The output mask */
            mask: components["schemas"]["ImageField"];
        };
        /**
         * FaceOff
         * @description Bound, extract, and mask a face from an image using MediaPipe detection
         */
        FaceOffInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Image for face detection
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Face Id
             * @description The face ID to process, numbered from 0. Multiple faces not supported. Find a face's ID with FaceIdentifier node.
             * @default 0
             */
            face_id?: number;
            /**
             * Minimum Confidence
             * @description Minimum confidence for face detection (lower if detection is failing)
             * @default 0.5
             */
            minimum_confidence?: number;
            /**
             * X Offset
             * @description X-axis offset of the mask
             * @default 0
             */
            x_offset?: number;
            /**
             * Y Offset
             * @description Y-axis offset of the mask
             * @default 0
             */
            y_offset?: number;
            /**
             * Padding
             * @description All-axis padding around the mask in pixels
             * @default 0
             */
            padding?: number;
            /**
             * Chunk
             * @description Whether to bypass full image face detection and default to image chunking. Chunking will occur if no faces are found in the full image.
             * @default false
             */
            chunk?: boolean;
            /**
             * type
             * @default face_off
             * @constant
             */
            type: "face_off";
        };
        /**
         * FaceOffOutput
         * @description Base class for FaceOff Output
         */
        FaceOffOutput: {
            /** @description The output image */
            image: components["schemas"]["ImageField"];
            /**
             * Width
             * @description The width of the image in pixels
             */
            width: number;
            /**
             * Height
             * @description The height of the image in pixels
             */
            height: number;
            /**
             * type
             * @default face_off_output
             * @constant
             */
            type: "face_off_output";
            /** @description The output mask */
            mask: components["schemas"]["ImageField"];
            /**
             * X
             * @description The x coordinate of the bounding box's left side
             */
            x: number;
            /**
             * Y
             * @description The y coordinate of the bounding box's top side
             */
            y: number;
        };
        /** FieldIdentifier */
        FieldIdentifier: {
            /**
             * Kind
             * @description The kind of field
             * @enum {string}
             */
            kind: "input" | "output";
            /**
             * Node Id
             * @description The ID of the node
             */
            node_id: string;
            /**
             * Field Name
             * @description The name of the field
             */
            field_name: string;
        };
        /**
         * FieldKind
         * @description The kind of field.
         *     - `Input`: An input field on a node.
         *     - `Output`: An output field on a node.
         *     - `Internal`: A field which is treated as an input, but cannot be used in node definitions. Metadata is
         *     one example. It is provided to nodes via the WithMetadata class, and we want to reserve the field name
         *     "metadata" for this on all nodes. `FieldKind` is used to short-circuit the field name validation logic,
         *     allowing "metadata" for that field.
         *     - `NodeAttribute`: The field is a node attribute. These are fields which are not inputs or outputs,
         *     but which are used to store information about the node. For example, the `id` and `type` fields are node
         *     attributes.
         *
         *     The presence of this in `json_schema_extra["field_kind"]` is used when initializing node schemas on app
         *     startup, and when generating the OpenAPI schema for the workflow editor.
         * @enum {string}
         */
        FieldKind: "input" | "output" | "internal" | "node_attribute";
        /**
         * Float Batch
         * @description Create a batched generation, where the workflow is executed once for each float in the batch.
         */
        FloatBatchInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Batch Group
             * @description The ID of this batch node's group. If provided, all batch nodes in with the same ID will be 'zipped' before execution, and all nodes' collections must be of the same size.
             * @default None
             * @enum {string}
             */
            batch_group_id?: "None" | "Group 1" | "Group 2" | "Group 3" | "Group 4" | "Group 5";
            /**
             * Floats
             * @description The floats to batch over
             * @default []
             */
            floats?: number[];
            /**
             * type
             * @default float_batch
             * @constant
             */
            type: "float_batch";
        };
        /**
         * Float Collection Primitive
         * @description A collection of float primitive values
         */
        FloatCollectionInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The collection of float values
             * @default []
             */
            collection?: number[];
            /**
             * type
             * @default float_collection
             * @constant
             */
            type: "float_collection";
        };
        /**
         * FloatCollectionOutput
         * @description Base class for nodes that output a collection of floats
         */
        FloatCollectionOutput: {
            /**
             * Collection
             * @description The float collection
             */
            collection: number[];
            /**
             * type
             * @default float_collection_output
             * @constant
             */
            type: "float_collection_output";
        };
        /**
         * Float Generator
         * @description Generated a range of floats for use in a batched generation
         */
        FloatGenerator: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Generator Type
             * @description The float generator.
             */
            generator: components["schemas"]["FloatGeneratorField"];
            /**
             * type
             * @default float_generator
             * @constant
             */
            type: "float_generator";
        };
        /** FloatGeneratorField */
        FloatGeneratorField: Record<string, never>;
        /**
         * FloatGeneratorOutput
         * @description Base class for nodes that output a collection of floats
         */
        FloatGeneratorOutput: {
            /**
             * Floats
             * @description The generated floats
             */
            floats: number[];
            /**
             * type
             * @default float_generator_output
             * @constant
             */
            type: "float_generator_output";
        };
        /**
         * Float Primitive
         * @description A float primitive value
         */
        FloatInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Value
             * @description The float value
             * @default 0
             */
            value?: number;
            /**
             * type
             * @default float
             * @constant
             */
            type: "float";
        };
        /**
         * Float Range
         * @description Creates a range
         */
        FloatLinearRangeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Start
             * @description The first value of the range
             * @default 5
             */
            start?: number;
            /**
             * Stop
             * @description The last value of the range
             * @default 10
             */
            stop?: number;
            /**
             * Steps
             * @description number of values to interpolate over (including start and stop)
             * @default 30
             */
            steps?: number;
            /**
             * type
             * @default float_range
             * @constant
             */
            type: "float_range";
        };
        /**
         * Float Math
         * @description Performs floating point math.
         */
        FloatMathInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Operation
             * @description The operation to perform
             * @default ADD
             * @enum {string}
             */
            operation?: "ADD" | "SUB" | "MUL" | "DIV" | "EXP" | "ABS" | "SQRT" | "MIN" | "MAX";
            /**
             * A
             * @description The first number
             * @default 1
             */
            a?: number;
            /**
             * B
             * @description The second number
             * @default 1
             */
            b?: number;
            /**
             * type
             * @default float_math
             * @constant
             */
            type: "float_math";
        };
        /**
         * FloatOutput
         * @description Base class for nodes that output a single float
         */
        FloatOutput: {
            /**
             * Value
             * @description The output float
             */
            value: number;
            /**
             * type
             * @default float_output
             * @constant
             */
            type: "float_output";
        };
        /**
         * Float To Integer
         * @description Rounds a float number to (a multiple of) an integer.
         */
        FloatToIntegerInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Value
             * @description The value to round
             * @default 0
             */
            value?: number;
            /**
             * Multiple of
             * @description The multiple to round to
             * @default 1
             */
            multiple?: number;
            /**
             * Method
             * @description The method to use for rounding
             * @default Nearest
             * @enum {string}
             */
            method?: "Nearest" | "Floor" | "Ceiling" | "Truncate";
            /**
             * type
             * @default float_to_int
             * @constant
             */
            type: "float_to_int";
        };
        /**
         * FluxConditioningField
         * @description A conditioning tensor primitive value
         */
        FluxConditioningField: {
            /**
             * Conditioning Name
             * @description The name of conditioning tensor
             */
            conditioning_name: string;
            /**
             * @description The mask associated with this conditioning tensor. Excluded regions should be set to False, included regions should be set to True.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
        };
        /**
         * FluxConditioningOutput
         * @description Base class for nodes that output a single conditioning tensor
         */
        FluxConditioningOutput: {
            /** @description Conditioning tensor */
            conditioning: components["schemas"]["FluxConditioningField"];
            /**
             * type
             * @default flux_conditioning_output
             * @constant
             */
            type: "flux_conditioning_output";
        };
        /**
         * Control LoRA - FLUX
         * @description LoRA model and Image to use with FLUX transformer generation.
         */
        FluxControlLoRALoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Control LoRA
             * @description Control LoRA model to load
             * @default null
             */
            lora?: components["schemas"]["ModelIdentifierField"];
            /**
             * @description The image to encode.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Weight
             * @description The weight of the LoRA.
             * @default 1
             */
            weight?: number;
            /**
             * type
             * @default flux_control_lora_loader
             * @constant
             */
            type: "flux_control_lora_loader";
        };
        /**
         * FluxControlLoRALoaderOutput
         * @description Flux Control LoRA Loader Output
         */
        FluxControlLoRALoaderOutput: {
            /**
             * Flux Control LoRA
             * @description Control LoRAs to apply on model loading
             * @default null
             */
            control_lora: components["schemas"]["ControlLoRAField"];
            /**
             * type
             * @default flux_control_lora_loader_output
             * @constant
             */
            type: "flux_control_lora_loader_output";
        };
        /** FluxControlNetField */
        FluxControlNetField: {
            /** @description The control image */
            image: components["schemas"]["ImageField"];
            /** @description The ControlNet model to use */
            control_model: components["schemas"]["ModelIdentifierField"];
            /**
             * Control Weight
             * @description The weight given to the ControlNet
             * @default 1
             */
            control_weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the ControlNet is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the ControlNet is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Resize Mode
             * @description The resize mode to use
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
            /**
             * Instantx Control Mode
             * @description The control mode for InstantX ControlNet union models. Ignored for other ControlNet models. The standard mapping is: canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6). Negative values will be treated as 'None'.
             * @default -1
             */
            instantx_control_mode?: number | null;
        };
        /**
         * FLUX ControlNet
         * @description Collect FLUX ControlNet info to pass to other nodes.
         */
        FluxControlNetInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The control image
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description ControlNet model to load
             * @default null
             */
            control_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Control Weight
             * @description The weight given to the ControlNet
             * @default 1
             */
            control_weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the ControlNet is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the ControlNet is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Resize Mode
             * @description The resize mode used
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
            /**
             * Instantx Control Mode
             * @description The control mode for InstantX ControlNet union models. Ignored for other ControlNet models. The standard mapping is: canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6). Negative values will be treated as 'None'.
             * @default -1
             */
            instantx_control_mode?: number | null;
            /**
             * type
             * @default flux_controlnet
             * @constant
             */
            type: "flux_controlnet";
        };
        /**
         * FluxControlNetOutput
         * @description FLUX ControlNet info
         */
        FluxControlNetOutput: {
            /** @description ControlNet(s) to apply */
            control: components["schemas"]["FluxControlNetField"];
            /**
             * type
             * @default flux_controlnet_output
             * @constant
             */
            type: "flux_controlnet_output";
        };
        /**
         * FLUX Denoise
         * @description Run denoising process with a FLUX transformer model.
         */
        FluxDenoiseInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"] | null;
            /**
             * @description A mask of the region to apply the denoising process to. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved.
             * @default null
             */
            denoise_mask?: components["schemas"]["DenoiseMaskField"] | null;
            /**
             * Denoising Start
             * @description When to start denoising, expressed a percentage of total steps
             * @default 0
             */
            denoising_start?: number;
            /**
             * Denoising End
             * @description When to stop denoising, expressed a percentage of total steps
             * @default 1
             */
            denoising_end?: number;
            /**
             * Add Noise
             * @description Add noise based on denoising start.
             * @default true
             */
            add_noise?: boolean;
            /**
             * Transformer
             * @description Flux model (Transformer) to load
             * @default null
             */
            transformer?: components["schemas"]["TransformerField"];
            /**
             * Control LoRA
             * @description Control LoRA model to load
             * @default null
             */
            control_lora?: components["schemas"]["ControlLoRAField"] | null;
            /**
             * Positive Text Conditioning
             * @description Positive conditioning tensor
             * @default null
             */
            positive_text_conditioning?: components["schemas"]["FluxConditioningField"] | components["schemas"]["FluxConditioningField"][];
            /**
             * Negative Text Conditioning
             * @description Negative conditioning tensor. Can be None if cfg_scale is 1.0.
             * @default null
             */
            negative_text_conditioning?: components["schemas"]["FluxConditioningField"] | components["schemas"]["FluxConditioningField"][] | null;
            /**
             * Redux Conditioning
             * @description FLUX Redux conditioning tensor.
             * @default null
             */
            redux_conditioning?: components["schemas"]["FluxReduxConditioningField"] | components["schemas"]["FluxReduxConditioningField"][] | null;
            /**
             * @description FLUX Fill conditioning.
             * @default null
             */
            fill_conditioning?: components["schemas"]["FluxFillConditioningField"] | null;
            /**
             * CFG Scale
             * @description Classifier-Free Guidance scale
             * @default 1
             */
            cfg_scale?: number | number[];
            /**
             * CFG Scale Start Step
             * @description Index of the first step to apply cfg_scale. Negative indices count backwards from the the last step (e.g. a value of -1 refers to the final step).
             * @default 0
             */
            cfg_scale_start_step?: number;
            /**
             * CFG Scale End Step
             * @description Index of the last step to apply cfg_scale. Negative indices count backwards from the last step (e.g. a value of -1 refers to the final step).
             * @default -1
             */
            cfg_scale_end_step?: number;
            /**
             * Width
             * @description Width of the generated image.
             * @default 1024
             */
            width?: number;
            /**
             * Height
             * @description Height of the generated image.
             * @default 1024
             */
            height?: number;
            /**
             * Num Steps
             * @description Number of diffusion steps. Recommended values are schnell: 4, dev: 50.
             * @default 4
             */
            num_steps?: number;
            /**
             * Guidance
             * @description The guidance strength. Higher values adhere more strictly to the prompt, and will produce less diverse images. FLUX dev only, ignored for schnell.
             * @default 4
             */
            guidance?: number;
            /**
             * Seed
             * @description Randomness seed for reproducibility.
             * @default 0
             */
            seed?: number;
            /**
             * Control
             * @description ControlNet models.
             * @default null
             */
            control?: components["schemas"]["FluxControlNetField"] | components["schemas"]["FluxControlNetField"][] | null;
            /**
             * @description VAE
             * @default null
             */
            controlnet_vae?: components["schemas"]["VAEField"] | null;
            /**
             * IP-Adapter
             * @description IP-Adapter to apply
             * @default null
             */
            ip_adapter?: components["schemas"]["IPAdapterField"] | components["schemas"]["IPAdapterField"][] | null;
            /**
             * type
             * @default flux_denoise
             * @constant
             */
            type: "flux_denoise";
        };
        /**
         * FLUX Denoise + Metadata
         * @description Run denoising process with a FLUX transformer model + metadata.
         */
        FluxDenoiseLatentsMetaInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"] | null;
            /**
             * @description A mask of the region to apply the denoising process to. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved.
             * @default null
             */
            denoise_mask?: components["schemas"]["DenoiseMaskField"] | null;
            /**
             * Denoising Start
             * @description When to start denoising, expressed a percentage of total steps
             * @default 0
             */
            denoising_start?: number;
            /**
             * Denoising End
             * @description When to stop denoising, expressed a percentage of total steps
             * @default 1
             */
            denoising_end?: number;
            /**
             * Add Noise
             * @description Add noise based on denoising start.
             * @default true
             */
            add_noise?: boolean;
            /**
             * Transformer
             * @description Flux model (Transformer) to load
             * @default null
             */
            transformer?: components["schemas"]["TransformerField"];
            /**
             * Control LoRA
             * @description Control LoRA model to load
             * @default null
             */
            control_lora?: components["schemas"]["ControlLoRAField"] | null;
            /**
             * Positive Text Conditioning
             * @description Positive conditioning tensor
             * @default null
             */
            positive_text_conditioning?: components["schemas"]["FluxConditioningField"] | components["schemas"]["FluxConditioningField"][];
            /**
             * Negative Text Conditioning
             * @description Negative conditioning tensor. Can be None if cfg_scale is 1.0.
             * @default null
             */
            negative_text_conditioning?: components["schemas"]["FluxConditioningField"] | components["schemas"]["FluxConditioningField"][] | null;
            /**
             * Redux Conditioning
             * @description FLUX Redux conditioning tensor.
             * @default null
             */
            redux_conditioning?: components["schemas"]["FluxReduxConditioningField"] | components["schemas"]["FluxReduxConditioningField"][] | null;
            /**
             * @description FLUX Fill conditioning.
             * @default null
             */
            fill_conditioning?: components["schemas"]["FluxFillConditioningField"] | null;
            /**
             * CFG Scale
             * @description Classifier-Free Guidance scale
             * @default 1
             */
            cfg_scale?: number | number[];
            /**
             * CFG Scale Start Step
             * @description Index of the first step to apply cfg_scale. Negative indices count backwards from the the last step (e.g. a value of -1 refers to the final step).
             * @default 0
             */
            cfg_scale_start_step?: number;
            /**
             * CFG Scale End Step
             * @description Index of the last step to apply cfg_scale. Negative indices count backwards from the last step (e.g. a value of -1 refers to the final step).
             * @default -1
             */
            cfg_scale_end_step?: number;
            /**
             * Width
             * @description Width of the generated image.
             * @default 1024
             */
            width?: number;
            /**
             * Height
             * @description Height of the generated image.
             * @default 1024
             */
            height?: number;
            /**
             * Num Steps
             * @description Number of diffusion steps. Recommended values are schnell: 4, dev: 50.
             * @default 4
             */
            num_steps?: number;
            /**
             * Guidance
             * @description The guidance strength. Higher values adhere more strictly to the prompt, and will produce less diverse images. FLUX dev only, ignored for schnell.
             * @default 4
             */
            guidance?: number;
            /**
             * Seed
             * @description Randomness seed for reproducibility.
             * @default 0
             */
            seed?: number;
            /**
             * Control
             * @description ControlNet models.
             * @default null
             */
            control?: components["schemas"]["FluxControlNetField"] | components["schemas"]["FluxControlNetField"][] | null;
            /**
             * @description VAE
             * @default null
             */
            controlnet_vae?: components["schemas"]["VAEField"] | null;
            /**
             * IP-Adapter
             * @description IP-Adapter to apply
             * @default null
             */
            ip_adapter?: components["schemas"]["IPAdapterField"] | components["schemas"]["IPAdapterField"][] | null;
            /**
             * type
             * @default flux_denoise_meta
             * @constant
             */
            type: "flux_denoise_meta";
        };
        /**
         * FluxFillConditioningField
         * @description A FLUX Fill conditioning field.
         */
        FluxFillConditioningField: {
            /** @description The FLUX Fill reference image. */
            image: components["schemas"]["ImageField"];
            /** @description The FLUX Fill inpaint mask. */
            mask: components["schemas"]["TensorField"];
        };
        /**
         * FLUX Fill Conditioning
         * @description Prepare the FLUX Fill conditioning data.
         */
        FluxFillInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The FLUX Fill reference image.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The bool inpainting mask. Excluded regions should be set to False, included regions should be set to True.
             * @default null
             */
            mask?: components["schemas"]["TensorField"];
            /**
             * type
             * @default flux_fill
             * @constant
             */
            type: "flux_fill";
        };
        /**
         * FluxFillOutput
         * @description The conditioning output of a FLUX Fill invocation.
         */
        FluxFillOutput: {
            /**
             * Conditioning
             * @description FLUX Redux conditioning tensor
             */
            fill_cond: components["schemas"]["FluxFillConditioningField"];
            /**
             * type
             * @default flux_fill_output
             * @constant
             */
            type: "flux_fill_output";
        };
        /**
         * FLUX IP-Adapter
         * @description Collects FLUX IP-Adapter info to pass to other nodes.
         */
        FluxIPAdapterInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The IP-Adapter image prompt(s).
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * IP-Adapter Model
             * @description The IP-Adapter model.
             * @default null
             */
            ip_adapter_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Clip Vision Model
             * @description CLIP Vision model to use.
             * @default ViT-L
             * @constant
             */
            clip_vision_model?: "ViT-L";
            /**
             * Weight
             * @description The weight given to the IP-Adapter
             * @default 1
             */
            weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the IP-Adapter is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the IP-Adapter is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * type
             * @default flux_ip_adapter
             * @constant
             */
            type: "flux_ip_adapter";
        };
        /**
         * Apply LoRA - FLUX
         * @description Apply a LoRA model to a FLUX transformer and/or text encoder.
         */
        FluxLoRALoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * LoRA
             * @description LoRA model to load
             * @default null
             */
            lora?: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight at which the LoRA is applied to each model
             * @default 0.75
             */
            weight?: number;
            /**
             * FLUX Transformer
             * @description Transformer
             * @default null
             */
            transformer?: components["schemas"]["TransformerField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * T5 Encoder
             * @description T5 tokenizer and text encoder
             * @default null
             */
            t5_encoder?: components["schemas"]["T5EncoderField"] | null;
            /**
             * type
             * @default flux_lora_loader
             * @constant
             */
            type: "flux_lora_loader";
        };
        /**
         * FluxLoRALoaderOutput
         * @description FLUX LoRA Loader Output
         */
        FluxLoRALoaderOutput: {
            /**
             * FLUX Transformer
             * @description Transformer
             * @default null
             */
            transformer: components["schemas"]["TransformerField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip: components["schemas"]["CLIPField"] | null;
            /**
             * T5 Encoder
             * @description T5 tokenizer and text encoder
             * @default null
             */
            t5_encoder: components["schemas"]["T5EncoderField"] | null;
            /**
             * type
             * @default flux_lora_loader_output
             * @constant
             */
            type: "flux_lora_loader_output";
        };
        /**
         * Main Model - FLUX
         * @description Loads a flux base model, outputting its submodels.
         */
        FluxModelLoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /** @description Flux model (Transformer) to load */
            model: components["schemas"]["ModelIdentifierField"];
            /**
             * T5 Encoder
             * @description T5 tokenizer and text encoder
             */
            t5_encoder_model: components["schemas"]["ModelIdentifierField"];
            /**
             * CLIP Embed
             * @description CLIP Embed loader
             */
            clip_embed_model: components["schemas"]["ModelIdentifierField"];
            /**
             * VAE
             * @description VAE model to load
             * @default null
             */
            vae_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default flux_model_loader
             * @constant
             */
            type: "flux_model_loader";
        };
        /**
         * FluxModelLoaderOutput
         * @description Flux base model loader output
         */
        FluxModelLoaderOutput: {
            /**
             * Transformer
             * @description Transformer
             */
            transformer: components["schemas"]["TransformerField"];
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip: components["schemas"]["CLIPField"];
            /**
             * T5 Encoder
             * @description T5 tokenizer and text encoder
             */
            t5_encoder: components["schemas"]["T5EncoderField"];
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * Max Seq Length
             * @description The max sequence length to used for the T5 encoder. (256 for schnell transformer, 512 for dev transformer)
             * @enum {integer}
             */
            max_seq_len: 256 | 512;
            /**
             * type
             * @default flux_model_loader_output
             * @constant
             */
            type: "flux_model_loader_output";
        };
        /**
         * FluxReduxConditioningField
         * @description A FLUX Redux conditioning tensor primitive value
         */
        FluxReduxConditioningField: {
            /** @description The Redux image conditioning tensor. */
            conditioning: components["schemas"]["TensorField"];
            /**
             * @description The mask associated with this conditioning tensor. Excluded regions should be set to False, included regions should be set to True.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
        };
        /**
         * FluxReduxConfig
         * @description Model config for FLUX Tools Redux model.
         */
        FluxReduxConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default flux_redux
             * @constant
             */
            type: "flux_redux";
            /**
             * Format
             * @default checkpoint
             * @constant
             */
            format: "checkpoint";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /**
         * FLUX Redux
         * @description Runs a FLUX Redux model to generate a conditioning tensor.
         */
        FluxReduxInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The FLUX Redux image prompt.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The bool mask associated with this FLUX Redux image prompt. Excluded regions should be set to False, included regions should be set to True.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
            /**
             * FLUX Redux Model
             * @description The FLUX Redux model to use.
             * @default null
             */
            redux_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Downsampling Factor
             * @description Redux Downsampling Factor (1-9)
             * @default 1
             */
            downsampling_factor?: number;
            /**
             * Downsampling Function
             * @description Redux Downsampling Function
             * @default area
             * @enum {string}
             */
            downsampling_function?: "nearest" | "bilinear" | "bicubic" | "area" | "nearest-exact";
            /**
             * Weight
             * @description Redux weight (0.0-1.0)
             * @default 1
             */
            weight?: number;
            /**
             * type
             * @default flux_redux
             * @constant
             */
            type: "flux_redux";
        };
        /**
         * FluxReduxOutput
         * @description The conditioning output of a FLUX Redux invocation.
         */
        FluxReduxOutput: {
            /**
             * Conditioning
             * @description FLUX Redux conditioning tensor
             */
            redux_cond: components["schemas"]["FluxReduxConditioningField"];
            /**
             * type
             * @default flux_redux_output
             * @constant
             */
            type: "flux_redux_output";
        };
        /**
         * Prompt - FLUX
         * @description Encodes and preps a prompt for a flux image.
         */
        FluxTextEncoderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"];
            /**
             * T5Encoder
             * @description T5 tokenizer and text encoder
             * @default null
             */
            t5_encoder?: components["schemas"]["T5EncoderField"];
            /**
             * T5 Max Seq Len
             * @description Max sequence length for the T5 encoder. Expected to be 256 for FLUX schnell models and 512 for FLUX dev models.
             * @default null
             * @enum {integer}
             */
            t5_max_seq_len?: 256 | 512;
            /**
             * Prompt
             * @description Text prompt to encode.
             * @default null
             */
            prompt?: string;
            /**
             * @description A mask defining the region that this conditioning prompt applies to.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
            /**
             * type
             * @default flux_text_encoder
             * @constant
             */
            type: "flux_text_encoder";
        };
        /**
         * Latents to Image - FLUX
         * @description Generates an image from latents.
         */
        FluxVaeDecodeInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * type
             * @default flux_vae_decode
             * @constant
             */
            type: "flux_vae_decode";
        };
        /**
         * Image to Latents - FLUX
         * @description Encodes an image into latents.
         */
        FluxVaeEncodeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to encode.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * type
             * @default flux_vae_encode
             * @constant
             */
            type: "flux_vae_encode";
        };
        /** FoundModel */
        FoundModel: {
            /**
             * Path
             * @description Path to the model
             */
            path: string;
            /**
             * Is Installed
             * @description Whether or not the model is already installed
             */
            is_installed: boolean;
        };
        /**
         * FreeUConfig
         * @description Configuration for the FreeU hyperparameters.
         *     - https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu
         *     - https://github.com/ChenyangSi/FreeU
         */
        FreeUConfig: {
            /**
             * S1
             * @description Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.
             */
            s1: number;
            /**
             * S2
             * @description Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.
             */
            s2: number;
            /**
             * B1
             * @description Scaling factor for stage 1 to amplify the contributions of backbone features.
             */
            b1: number;
            /**
             * B2
             * @description Scaling factor for stage 2 to amplify the contributions of backbone features.
             */
            b2: number;
        };
        /**
         * Apply FreeU - SD1.5, SDXL
         * @description Applies FreeU to the UNet. Suggested values (b1/b2/s1/s2):
         *
         *     SD1.5: 1.2/1.4/0.9/0.2,
         *     SD2: 1.1/1.2/0.9/0.2,
         *     SDXL: 1.1/1.2/0.6/0.4,
         */
        FreeUInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"];
            /**
             * B1
             * @description Scaling factor for stage 1 to amplify the contributions of backbone features.
             * @default 1.2
             */
            b1?: number;
            /**
             * B2
             * @description Scaling factor for stage 2 to amplify the contributions of backbone features.
             * @default 1.4
             */
            b2?: number;
            /**
             * S1
             * @description Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.
             * @default 0.9
             */
            s1?: number;
            /**
             * S2
             * @description Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.
             * @default 0.2
             */
            s2?: number;
            /**
             * type
             * @default freeu
             * @constant
             */
            type: "freeu";
        };
        /**
         * Get Image Mask Bounding Box
         * @description Gets the bounding box of the given mask image.
         */
        GetMaskBoundingBoxInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The mask to crop.
             * @default null
             */
            mask?: components["schemas"]["ImageField"];
            /**
             * Margin
             * @description Margin to add to the bounding box.
             * @default 0
             */
            margin?: number;
            /**
             * @description Color of the mask in the image.
             * @default {
             *       "r": 255,
             *       "g": 255,
             *       "b": 255,
             *       "a": 255
             *     }
             */
            mask_color?: components["schemas"]["ColorField"];
            /**
             * type
             * @default get_image_mask_bounding_box
             * @constant
             */
            type: "get_image_mask_bounding_box";
        };
        /** GlmEncoderField */
        GlmEncoderField: {
            /** @description Info to load tokenizer submodel */
            tokenizer: components["schemas"]["ModelIdentifierField"];
            /** @description Info to load text_encoder submodel */
            text_encoder: components["schemas"]["ModelIdentifierField"];
        };
        /**
         * GradientMaskOutput
         * @description Outputs a denoise mask and an image representing the total gradient of the mask.
         */
        GradientMaskOutput: {
            /** @description Mask for denoise model run. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved. */
            denoise_mask: components["schemas"]["DenoiseMaskField"];
            /** @description Image representing the total gradient area of the mask. For paste-back purposes. */
            expanded_mask_area: components["schemas"]["ImageField"];
            /**
             * type
             * @default gradient_mask_output
             * @constant
             */
            type: "gradient_mask_output";
        };
        /** Graph */
        Graph: {
            /**
             * Id
             * @description The id of this graph
             */
            id?: string;
            /**
             * Nodes
             * @description The nodes in this graph
             */
            nodes?: {
                [key: string]: components["schemas"]["AddInvocation"] | components["schemas"]["AlphaMaskToTensorInvocation"] | components["schemas"]["ApplyMaskTensorToImageInvocation"] | components["schemas"]["ApplyMaskToImageInvocation"] | components["schemas"]["BlankImageInvocation"] | components["schemas"]["BlendLatentsInvocation"] | components["schemas"]["BooleanCollectionInvocation"] | components["schemas"]["BooleanInvocation"] | components["schemas"]["BoundingBoxInvocation"] | components["schemas"]["CLIPSkipInvocation"] | components["schemas"]["CV2InfillInvocation"] | components["schemas"]["CalculateImageTilesEvenSplitInvocation"] | components["schemas"]["CalculateImageTilesInvocation"] | components["schemas"]["CalculateImageTilesMinimumOverlapInvocation"] | components["schemas"]["CannyEdgeDetectionInvocation"] | components["schemas"]["CanvasPasteBackInvocation"] | components["schemas"]["CanvasV2MaskAndCropInvocation"] | components["schemas"]["CenterPadCropInvocation"] | components["schemas"]["CogView4DenoiseInvocation"] | components["schemas"]["CogView4ImageToLatentsInvocation"] | components["schemas"]["CogView4LatentsToImageInvocation"] | components["schemas"]["CogView4ModelLoaderInvocation"] | components["schemas"]["CogView4TextEncoderInvocation"] | components["schemas"]["CollectInvocation"] | components["schemas"]["ColorCorrectInvocation"] | components["schemas"]["ColorInvocation"] | components["schemas"]["ColorMapInvocation"] | components["schemas"]["CompelInvocation"] | components["schemas"]["ConditioningCollectionInvocation"] | components["schemas"]["ConditioningInvocation"] | components["schemas"]["ContentShuffleInvocation"] | components["schemas"]["ControlNetInvocation"] | components["schemas"]["CoreMetadataInvocation"] | components["schemas"]["CreateDenoiseMaskInvocation"] | components["schemas"]["CreateGradientMaskInvocation"] | components["schemas"]["CropImageToBoundingBoxInvocation"] | components["schemas"]["CropLatentsCoreInvocation"] | components["schemas"]["CvInpaintInvocation"] | components["schemas"]["DWOpenposeDetectionInvocation"] | components["schemas"]["DenoiseLatentsInvocation"] | components["schemas"]["DenoiseLatentsMetaInvocation"] | components["schemas"]["DepthAnythingDepthEstimationInvocation"] | components["schemas"]["DivideInvocation"] | components["schemas"]["DynamicPromptInvocation"] | components["schemas"]["ESRGANInvocation"] | components["schemas"]["ExpandMaskWithFadeInvocation"] | components["schemas"]["FLUXLoRACollectionLoader"] | components["schemas"]["FaceIdentifierInvocation"] | components["schemas"]["FaceMaskInvocation"] | components["schemas"]["FaceOffInvocation"] | components["schemas"]["FloatBatchInvocation"] | components["schemas"]["FloatCollectionInvocation"] | components["schemas"]["FloatGenerator"] | components["schemas"]["FloatInvocation"] | components["schemas"]["FloatLinearRangeInvocation"] | components["schemas"]["FloatMathInvocation"] | components["schemas"]["FloatToIntegerInvocation"] | components["schemas"]["FluxControlLoRALoaderInvocation"] | components["schemas"]["FluxControlNetInvocation"] | components["schemas"]["FluxDenoiseInvocation"] | components["schemas"]["FluxDenoiseLatentsMetaInvocation"] | components["schemas"]["FluxFillInvocation"] | components["schemas"]["FluxIPAdapterInvocation"] | components["schemas"]["FluxLoRALoaderInvocation"] | components["schemas"]["FluxModelLoaderInvocation"] | components["schemas"]["FluxReduxInvocation"] | components["schemas"]["FluxTextEncoderInvocation"] | components["schemas"]["FluxVaeDecodeInvocation"] | components["schemas"]["FluxVaeEncodeInvocation"] | components["schemas"]["FreeUInvocation"] | components["schemas"]["GetMaskBoundingBoxInvocation"] | components["schemas"]["GroundingDinoInvocation"] | components["schemas"]["HEDEdgeDetectionInvocation"] | components["schemas"]["HeuristicResizeInvocation"] | components["schemas"]["IPAdapterInvocation"] | components["schemas"]["IdealSizeInvocation"] | components["schemas"]["ImageBatchInvocation"] | components["schemas"]["ImageBlurInvocation"] | components["schemas"]["ImageChannelInvocation"] | components["schemas"]["ImageChannelMultiplyInvocation"] | components["schemas"]["ImageChannelOffsetInvocation"] | components["schemas"]["ImageCollectionInvocation"] | components["schemas"]["ImageConvertInvocation"] | components["schemas"]["ImageCropInvocation"] | components["schemas"]["ImageGenerator"] | components["schemas"]["ImageHueAdjustmentInvocation"] | components["schemas"]["ImageInverseLerpInvocation"] | components["schemas"]["ImageInvocation"] | components["schemas"]["ImageLerpInvocation"] | components["schemas"]["ImageMaskToTensorInvocation"] | components["schemas"]["ImageMultiplyInvocation"] | components["schemas"]["ImageNSFWBlurInvocation"] | components["schemas"]["ImageNoiseInvocation"] | components["schemas"]["ImagePanelLayoutInvocation"] | components["schemas"]["ImagePasteInvocation"] | components["schemas"]["ImageResizeInvocation"] | components["schemas"]["ImageScaleInvocation"] | components["schemas"]["ImageToLatentsInvocation"] | components["schemas"]["ImageWatermarkInvocation"] | components["schemas"]["InfillColorInvocation"] | components["schemas"]["InfillPatchMatchInvocation"] | components["schemas"]["InfillTileInvocation"] | components["schemas"]["IntegerBatchInvocation"] | components["schemas"]["IntegerCollectionInvocation"] | components["schemas"]["IntegerGenerator"] | components["schemas"]["IntegerInvocation"] | components["schemas"]["IntegerMathInvocation"] | components["schemas"]["InvertTensorMaskInvocation"] | components["schemas"]["InvokeAdjustImageHuePlusInvocation"] | components["schemas"]["InvokeEquivalentAchromaticLightnessInvocation"] | components["schemas"]["InvokeImageBlendInvocation"] | components["schemas"]["InvokeImageCompositorInvocation"] | components["schemas"]["InvokeImageDilateOrErodeInvocation"] | components["schemas"]["InvokeImageEnhanceInvocation"] | components["schemas"]["InvokeImageValueThresholdsInvocation"] | components["schemas"]["IterateInvocation"] | components["schemas"]["LaMaInfillInvocation"] | components["schemas"]["LatentsCollectionInvocation"] | components["schemas"]["LatentsInvocation"] | components["schemas"]["LatentsToImageInvocation"] | components["schemas"]["LineartAnimeEdgeDetectionInvocation"] | components["schemas"]["LineartEdgeDetectionInvocation"] | components["schemas"]["LlavaOnevisionVllmInvocation"] | components["schemas"]["LoRACollectionLoader"] | components["schemas"]["LoRALoaderInvocation"] | components["schemas"]["LoRASelectorInvocation"] | components["schemas"]["MLSDDetectionInvocation"] | components["schemas"]["MainModelLoaderInvocation"] | components["schemas"]["MaskCombineInvocation"] | components["schemas"]["MaskEdgeInvocation"] | components["schemas"]["MaskFromAlphaInvocation"] | components["schemas"]["MaskFromIDInvocation"] | components["schemas"]["MaskTensorToImageInvocation"] | components["schemas"]["MediaPipeFaceDetectionInvocation"] | components["schemas"]["MergeMetadataInvocation"] | components["schemas"]["MergeTilesToImageInvocation"] | components["schemas"]["MetadataFieldExtractorInvocation"] | components["schemas"]["MetadataFromImageInvocation"] | components["schemas"]["MetadataInvocation"] | components["schemas"]["MetadataItemInvocation"] | components["schemas"]["MetadataItemLinkedInvocation"] | components["schemas"]["MetadataToBoolInvocation"] | components["schemas"]["MetadataToControlnetsInvocation"] | components["schemas"]["MetadataToFloatInvocation"] | components["schemas"]["MetadataToIPAdaptersInvocation"] | components["schemas"]["MetadataToIntegerInvocation"] | components["schemas"]["MetadataToLorasCollectionInvocation"] | components["schemas"]["MetadataToLorasInvocation"] | components["schemas"]["MetadataToModelInvocation"] | components["schemas"]["MetadataToSDXLLorasInvocation"] | components["schemas"]["MetadataToSDXLModelInvocation"] | components["schemas"]["MetadataToSchedulerInvocation"] | components["schemas"]["MetadataToStringInvocation"] | components["schemas"]["MetadataToT2IAdaptersInvocation"] | components["schemas"]["MetadataToVAEInvocation"] | components["schemas"]["ModelIdentifierInvocation"] | components["schemas"]["MultiplyInvocation"] | components["schemas"]["NoiseInvocation"] | components["schemas"]["NormalMapInvocation"] | components["schemas"]["PairTileImageInvocation"] | components["schemas"]["PasteImageIntoBoundingBoxInvocation"] | components["schemas"]["PiDiNetEdgeDetectionInvocation"] | components["schemas"]["PromptsFromFileInvocation"] | components["schemas"]["RandomFloatInvocation"] | components["schemas"]["RandomIntInvocation"] | components["schemas"]["RandomRangeInvocation"] | components["schemas"]["RangeInvocation"] | components["schemas"]["RangeOfSizeInvocation"] | components["schemas"]["RectangleMaskInvocation"] | components["schemas"]["ResizeLatentsInvocation"] | components["schemas"]["RoundInvocation"] | components["schemas"]["SD3DenoiseInvocation"] | components["schemas"]["SD3ImageToLatentsInvocation"] | components["schemas"]["SD3LatentsToImageInvocation"] | components["schemas"]["SDXLCompelPromptInvocation"] | components["schemas"]["SDXLLoRACollectionLoader"] | components["schemas"]["SDXLLoRALoaderInvocation"] | components["schemas"]["SDXLModelLoaderInvocation"] | components["schemas"]["SDXLRefinerCompelPromptInvocation"] | components["schemas"]["SDXLRefinerModelLoaderInvocation"] | components["schemas"]["SaveImageInvocation"] | components["schemas"]["ScaleLatentsInvocation"] | components["schemas"]["SchedulerInvocation"] | components["schemas"]["Sd3ModelLoaderInvocation"] | components["schemas"]["Sd3TextEncoderInvocation"] | components["schemas"]["SeamlessModeInvocation"] | components["schemas"]["SegmentAnythingInvocation"] | components["schemas"]["ShowImageInvocation"] | components["schemas"]["SpandrelImageToImageAutoscaleInvocation"] | components["schemas"]["SpandrelImageToImageInvocation"] | components["schemas"]["StringBatchInvocation"] | components["schemas"]["StringCollectionInvocation"] | components["schemas"]["StringGenerator"] | components["schemas"]["StringInvocation"] | components["schemas"]["StringJoinInvocation"] | components["schemas"]["StringJoinThreeInvocation"] | components["schemas"]["StringReplaceInvocation"] | components["schemas"]["StringSplitInvocation"] | components["schemas"]["StringSplitNegInvocation"] | components["schemas"]["SubtractInvocation"] | components["schemas"]["T2IAdapterInvocation"] | components["schemas"]["TileToPropertiesInvocation"] | components["schemas"]["TiledMultiDiffusionDenoiseLatents"] | components["schemas"]["UnsharpMaskInvocation"] | components["schemas"]["VAELoaderInvocation"];
            };
            /**
             * Edges
             * @description The connections between nodes and their fields in this graph
             */
            edges?: components["schemas"]["Edge"][];
        };
        /**
         * GraphExecutionState
         * @description Tracks the state of a graph execution
         */
        GraphExecutionState: {
            /**
             * Id
             * @description The id of the execution state
             */
            id?: string;
            /** @description The graph being executed */
            graph: components["schemas"]["Graph"];
            /** @description The expanded graph of activated and executed nodes */
            execution_graph?: components["schemas"]["Graph"];
            /**
             * Executed
             * @description The set of node ids that have been executed
             */
            executed?: string[];
            /**
             * Executed History
             * @description The list of node ids that have been executed, in order of execution
             */
            executed_history?: string[];
            /**
             * Results
             * @description The results of node executions
             */
            results?: {
                [key: string]: components["schemas"]["BooleanCollectionOutput"] | components["schemas"]["BooleanOutput"] | components["schemas"]["BoundingBoxCollectionOutput"] | components["schemas"]["BoundingBoxOutput"] | components["schemas"]["CLIPOutput"] | components["schemas"]["CLIPSkipInvocationOutput"] | components["schemas"]["CalculateImageTilesOutput"] | components["schemas"]["CogView4ConditioningOutput"] | components["schemas"]["CogView4ModelLoaderOutput"] | components["schemas"]["CollectInvocationOutput"] | components["schemas"]["ColorCollectionOutput"] | components["schemas"]["ColorOutput"] | components["schemas"]["ConditioningCollectionOutput"] | components["schemas"]["ConditioningOutput"] | components["schemas"]["ControlOutput"] | components["schemas"]["DenoiseMaskOutput"] | components["schemas"]["FaceMaskOutput"] | components["schemas"]["FaceOffOutput"] | components["schemas"]["FloatCollectionOutput"] | components["schemas"]["FloatGeneratorOutput"] | components["schemas"]["FloatOutput"] | components["schemas"]["FluxConditioningOutput"] | components["schemas"]["FluxControlLoRALoaderOutput"] | components["schemas"]["FluxControlNetOutput"] | components["schemas"]["FluxFillOutput"] | components["schemas"]["FluxLoRALoaderOutput"] | components["schemas"]["FluxModelLoaderOutput"] | components["schemas"]["FluxReduxOutput"] | components["schemas"]["GradientMaskOutput"] | components["schemas"]["IPAdapterOutput"] | components["schemas"]["IdealSizeOutput"] | components["schemas"]["ImageCollectionOutput"] | components["schemas"]["ImageGeneratorOutput"] | components["schemas"]["ImageOutput"] | components["schemas"]["ImagePanelCoordinateOutput"] | components["schemas"]["IntegerCollectionOutput"] | components["schemas"]["IntegerGeneratorOutput"] | components["schemas"]["IntegerOutput"] | components["schemas"]["IterateInvocationOutput"] | components["schemas"]["LatentsCollectionOutput"] | components["schemas"]["LatentsMetaOutput"] | components["schemas"]["LatentsOutput"] | components["schemas"]["LoRALoaderOutput"] | components["schemas"]["LoRASelectorOutput"] | components["schemas"]["MDControlListOutput"] | components["schemas"]["MDIPAdapterListOutput"] | components["schemas"]["MDT2IAdapterListOutput"] | components["schemas"]["MaskOutput"] | components["schemas"]["MetadataItemOutput"] | components["schemas"]["MetadataOutput"] | components["schemas"]["MetadataToLorasCollectionOutput"] | components["schemas"]["MetadataToModelOutput"] | components["schemas"]["MetadataToSDXLModelOutput"] | components["schemas"]["ModelIdentifierOutput"] | components["schemas"]["ModelLoaderOutput"] | components["schemas"]["NoiseOutput"] | components["schemas"]["PairTileImageOutput"] | components["schemas"]["SD3ConditioningOutput"] | components["schemas"]["SDXLLoRALoaderOutput"] | components["schemas"]["SDXLModelLoaderOutput"] | components["schemas"]["SDXLRefinerModelLoaderOutput"] | components["schemas"]["SchedulerOutput"] | components["schemas"]["Sd3ModelLoaderOutput"] | components["schemas"]["SeamlessModeOutput"] | components["schemas"]["String2Output"] | components["schemas"]["StringCollectionOutput"] | components["schemas"]["StringGeneratorOutput"] | components["schemas"]["StringOutput"] | components["schemas"]["StringPosNegOutput"] | components["schemas"]["T2IAdapterOutput"] | components["schemas"]["TileToPropertiesOutput"] | components["schemas"]["UNetOutput"] | components["schemas"]["VAEOutput"];
            };
            /**
             * Errors
             * @description Errors raised when executing nodes
             */
            errors?: {
                [key: string]: string;
            };
            /**
             * Prepared Source Mapping
             * @description The map of prepared nodes to original graph nodes
             */
            prepared_source_mapping?: {
                [key: string]: string;
            };
            /**
             * Source Prepared Mapping
             * @description The map of original graph nodes to prepared nodes
             */
            source_prepared_mapping?: {
                [key: string]: string[];
            };
        };
        /**
         * Grounding DINO (Text Prompt Object Detection)
         * @description Runs a Grounding DINO model. Performs zero-shot bounding-box object detection from a text prompt.
         */
        GroundingDinoInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Model
             * @description The Grounding DINO model to use.
             * @default null
             * @enum {string}
             */
            model?: "grounding-dino-tiny" | "grounding-dino-base";
            /**
             * Prompt
             * @description The prompt describing the object to segment.
             * @default null
             */
            prompt?: string;
            /**
             * @description The image to segment.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Detection Threshold
             * @description The detection threshold for the Grounding DINO model. All detected bounding boxes with scores above this threshold will be returned.
             * @default 0.3
             */
            detection_threshold?: number;
            /**
             * type
             * @default grounding_dino
             * @constant
             */
            type: "grounding_dino";
        };
        /**
         * HED Edge Detection
         * @description Geneartes an edge map using the HED (softedge) model.
         */
        HEDEdgeDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Scribble
             * @description Whether or not to use scribble mode
             * @default false
             */
            scribble?: boolean;
            /**
             * type
             * @default hed_edge_detection
             * @constant
             */
            type: "hed_edge_detection";
        };
        /**
         * HFModelSource
         * @description A HuggingFace repo_id with optional variant, sub-folder and access token.
         *     Note that the variant option, if not provided to the constructor, will default to fp16, which is
         *     what people (almost) always want.
         */
        HFModelSource: {
            /** Repo Id */
            repo_id: string;
            /** @default fp16 */
            variant?: components["schemas"]["ModelRepoVariant"] | null;
            /** Subfolder */
            subfolder?: string | null;
            /** Access Token */
            access_token?: string | null;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "hf";
        };
        /**
         * HFTokenStatus
         * @enum {string}
         */
        HFTokenStatus: "valid" | "invalid" | "unknown";
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /**
         * Heuristic Resize
         * @description Resize an image using a heuristic method. Preserves edge maps.
         */
        HeuristicResizeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to resize
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Width
             * @description The width to resize to (px)
             * @default 512
             */
            width?: number;
            /**
             * Height
             * @description The height to resize to (px)
             * @default 512
             */
            height?: number;
            /**
             * type
             * @default heuristic_resize
             * @constant
             */
            type: "heuristic_resize";
        };
        /**
         * HuggingFaceMetadata
         * @description Extended metadata fields provided by HuggingFace.
         */
        HuggingFaceMetadata: {
            /**
             * Name
             * @description model's name
             */
            name: string;
            /**
             * Files
             * @description model files and their sizes
             */
            files?: components["schemas"]["RemoteModelFile"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "huggingface";
            /**
             * Id
             * @description The HF model id
             */
            id: string;
            /**
             * Api Response
             * @description Response from the HF API as stringified JSON
             */
            api_response?: string | null;
            /**
             * Is Diffusers
             * @description Whether the metadata is for a Diffusers format model
             * @default false
             */
            is_diffusers?: boolean;
            /**
             * Ckpt Urls
             * @description URLs for all checkpoint format models in the metadata
             */
            ckpt_urls?: string[] | null;
        };
        /** HuggingFaceModels */
        HuggingFaceModels: {
            /**
             * Urls
             * @description URLs for all checkpoint format models in the metadata
             */
            urls: string[] | null;
            /**
             * Is Diffusers
             * @description Whether the metadata is for a Diffusers format model
             */
            is_diffusers: boolean;
        };
        /**
         * IPAdapterCheckpointConfig
         * @description Model config for IP Adapter checkpoint format models.
         */
        IPAdapterCheckpointConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default ip_adapter
             * @constant
             */
            type: "ip_adapter";
            /**
             * Format
             * @default checkpoint
             * @constant
             */
            format: "checkpoint";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /** IPAdapterField */
        IPAdapterField: {
            /**
             * Image
             * @description The IP-Adapter image prompt(s).
             */
            image: components["schemas"]["ImageField"] | components["schemas"]["ImageField"][];
            /** @description The IP-Adapter model to use. */
            ip_adapter_model: components["schemas"]["ModelIdentifierField"];
            /** @description The name of the CLIP image encoder model. */
            image_encoder_model: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight given to the IP-Adapter.
             * @default 1
             */
            weight?: number | number[];
            /**
             * Target Blocks
             * @description The IP Adapter blocks to apply
             * @default []
             */
            target_blocks?: string[];
            /**
             * Begin Step Percent
             * @description When the IP-Adapter is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the IP-Adapter is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * @description The bool mask associated with this IP-Adapter. Excluded regions should be set to False, included regions should be set to True.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
        };
        /**
         * IP-Adapter - SD1.5, SDXL
         * @description Collects IP-Adapter info to pass to other nodes.
         */
        IPAdapterInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Image
             * @description The IP-Adapter image prompt(s).
             * @default null
             */
            image?: components["schemas"]["ImageField"] | components["schemas"]["ImageField"][];
            /**
             * IP-Adapter Model
             * @description The IP-Adapter model.
             * @default null
             */
            ip_adapter_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Clip Vision Model
             * @description CLIP Vision model to use. Overrides model settings. Mandatory for checkpoint models.
             * @default ViT-H
             * @enum {string}
             */
            clip_vision_model?: "ViT-H" | "ViT-G" | "ViT-L";
            /**
             * Weight
             * @description The weight given to the IP-Adapter
             * @default 1
             */
            weight?: number | number[];
            /**
             * Method
             * @description The method to apply the IP-Adapter
             * @default full
             * @enum {string}
             */
            method?: "full" | "style" | "composition";
            /**
             * Begin Step Percent
             * @description When the IP-Adapter is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the IP-Adapter is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * @description A mask defining the region that this IP-Adapter applies to.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
            /**
             * type
             * @default ip_adapter
             * @constant
             */
            type: "ip_adapter";
        };
        /**
         * IPAdapterInvokeAIConfig
         * @description Model config for IP Adapter diffusers format models.
         */
        IPAdapterInvokeAIConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default ip_adapter
             * @constant
             */
            type: "ip_adapter";
            /**
             * Format
             * @default invokeai
             * @constant
             */
            format: "invokeai";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** Image Encoder Model Id */
            image_encoder_model_id: string;
        };
        /**
         * IPAdapterMetadataField
         * @description IP Adapter Field, minus the CLIP Vision Encoder model
         */
        IPAdapterMetadataField: {
            /** @description The IP-Adapter image prompt. */
            image: components["schemas"]["ImageField"];
            /** @description The IP-Adapter model. */
            ip_adapter_model: components["schemas"]["ModelIdentifierField"];
            /**
             * Clip Vision Model
             * @description The CLIP Vision model
             * @enum {string}
             */
            clip_vision_model: "ViT-L" | "ViT-H" | "ViT-G";
            /**
             * Method
             * @description Method to apply IP Weights with
             * @enum {string}
             */
            method: "full" | "style" | "composition";
            /**
             * Weight
             * @description The weight given to the IP-Adapter
             */
            weight: number | number[];
            /**
             * Begin Step Percent
             * @description When the IP-Adapter is first applied (% of total steps)
             */
            begin_step_percent: number;
            /**
             * End Step Percent
             * @description When the IP-Adapter is last applied (% of total steps)
             */
            end_step_percent: number;
        };
        /** IPAdapterOutput */
        IPAdapterOutput: {
            /**
             * IP-Adapter
             * @description IP-Adapter to apply
             */
            ip_adapter: components["schemas"]["IPAdapterField"];
            /**
             * type
             * @default ip_adapter_output
             * @constant
             */
            type: "ip_adapter_output";
        };
        /**
         * Ideal Size - SD1.5, SDXL
         * @description Calculates the ideal size for generation to avoid duplication
         */
        IdealSizeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Width
             * @description Final image width
             * @default 1024
             */
            width?: number;
            /**
             * Height
             * @description Final image height
             * @default 576
             */
            height?: number;
            /**
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"];
            /**
             * Multiplier
             * @description Amount to multiply the model's dimensions by when calculating the ideal size (may result in initial generation artifacts if too large)
             * @default 1
             */
            multiplier?: number;
            /**
             * type
             * @default ideal_size
             * @constant
             */
            type: "ideal_size";
        };
        /**
         * IdealSizeOutput
         * @description Base class for invocations that output an image
         */
        IdealSizeOutput: {
            /**
             * Width
             * @description The ideal width of the image (in pixels)
             */
            width: number;
            /**
             * Height
             * @description The ideal height of the image (in pixels)
             */
            height: number;
            /**
             * type
             * @default ideal_size_output
             * @constant
             */
            type: "ideal_size_output";
        };
        /**
         * Image Batch
         * @description Create a batched generation, where the workflow is executed once for each image in the batch.
         */
        ImageBatchInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Batch Group
             * @description The ID of this batch node's group. If provided, all batch nodes in with the same ID will be 'zipped' before execution, and all nodes' collections must be of the same size.
             * @default None
             * @enum {string}
             */
            batch_group_id?: "None" | "Group 1" | "Group 2" | "Group 3" | "Group 4" | "Group 5";
            /**
             * Images
             * @description The images to batch over
             * @default []
             */
            images?: components["schemas"]["ImageField"][];
            /**
             * type
             * @default image_batch
             * @constant
             */
            type: "image_batch";
        };
        /**
         * Blur Image
         * @description Blurs an image
         */
        ImageBlurInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to blur
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Radius
             * @description The blur radius
             * @default 8
             */
            radius?: number;
            /**
             * Blur Type
             * @description The type of blur
             * @default gaussian
             * @enum {string}
             */
            blur_type?: "gaussian" | "box";
            /**
             * type
             * @default img_blur
             * @constant
             */
            type: "img_blur";
        };
        /**
         * ImageCategory
         * @description The category of an image.
         *
         *     - GENERAL: The image is an output, init image, or otherwise an image without a specialized purpose.
         *     - MASK: The image is a mask image.
         *     - CONTROL: The image is a ControlNet control image.
         *     - USER: The image is a user-provide image.
         *     - OTHER: The image is some other type of image with a specialized purpose. To be used by external nodes.
         * @enum {string}
         */
        ImageCategory: "general" | "mask" | "control" | "user" | "other";
        /**
         * Extract Image Channel
         * @description Gets a channel from an image.
         */
        ImageChannelInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to get the channel from
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Channel
             * @description The channel to get
             * @default A
             * @enum {string}
             */
            channel?: "A" | "R" | "G" | "B";
            /**
             * type
             * @default img_chan
             * @constant
             */
            type: "img_chan";
        };
        /**
         * Multiply Image Channel
         * @description Scale a specific color channel of an image.
         */
        ImageChannelMultiplyInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to adjust
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Channel
             * @description Which channel to adjust
             * @default null
             * @enum {string}
             */
            channel?: "Red (RGBA)" | "Green (RGBA)" | "Blue (RGBA)" | "Alpha (RGBA)" | "Cyan (CMYK)" | "Magenta (CMYK)" | "Yellow (CMYK)" | "Black (CMYK)" | "Hue (HSV)" | "Saturation (HSV)" | "Value (HSV)" | "Luminosity (LAB)" | "A (LAB)" | "B (LAB)" | "Y (YCbCr)" | "Cb (YCbCr)" | "Cr (YCbCr)";
            /**
             * Scale
             * @description The amount to scale the channel by.
             * @default 1
             */
            scale?: number;
            /**
             * Invert Channel
             * @description Invert the channel after scaling
             * @default false
             */
            invert_channel?: boolean;
            /**
             * type
             * @default img_channel_multiply
             * @constant
             */
            type: "img_channel_multiply";
        };
        /**
         * Offset Image Channel
         * @description Add or subtract a value from a specific color channel of an image.
         */
        ImageChannelOffsetInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to adjust
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Channel
             * @description Which channel to adjust
             * @default null
             * @enum {string}
             */
            channel?: "Red (RGBA)" | "Green (RGBA)" | "Blue (RGBA)" | "Alpha (RGBA)" | "Cyan (CMYK)" | "Magenta (CMYK)" | "Yellow (CMYK)" | "Black (CMYK)" | "Hue (HSV)" | "Saturation (HSV)" | "Value (HSV)" | "Luminosity (LAB)" | "A (LAB)" | "B (LAB)" | "Y (YCbCr)" | "Cb (YCbCr)" | "Cr (YCbCr)";
            /**
             * Offset
             * @description The amount to adjust the channel by
             * @default 0
             */
            offset?: number;
            /**
             * type
             * @default img_channel_offset
             * @constant
             */
            type: "img_channel_offset";
        };
        /**
         * Image Collection Primitive
         * @description A collection of image primitive values
         */
        ImageCollectionInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The collection of image values
             * @default null
             */
            collection?: components["schemas"]["ImageField"][];
            /**
             * type
             * @default image_collection
             * @constant
             */
            type: "image_collection";
        };
        /**
         * ImageCollectionOutput
         * @description Base class for nodes that output a collection of images
         */
        ImageCollectionOutput: {
            /**
             * Collection
             * @description The output images
             */
            collection: components["schemas"]["ImageField"][];
            /**
             * type
             * @default image_collection_output
             * @constant
             */
            type: "image_collection_output";
        };
        /**
         * Convert Image Mode
         * @description Converts an image to a different mode.
         */
        ImageConvertInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to convert
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Mode
             * @description The mode to convert to
             * @default L
             * @enum {string}
             */
            mode?: "L" | "RGB" | "RGBA" | "CMYK" | "YCbCr" | "LAB" | "HSV" | "I" | "F";
            /**
             * type
             * @default img_conv
             * @constant
             */
            type: "img_conv";
        };
        /**
         * Crop Image
         * @description Crops an image to a specified box. The box can be outside of the image.
         */
        ImageCropInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to crop
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * X
             * @description The left x coordinate of the crop rectangle
             * @default 0
             */
            x?: number;
            /**
             * Y
             * @description The top y coordinate of the crop rectangle
             * @default 0
             */
            y?: number;
            /**
             * Width
             * @description The width of the crop rectangle
             * @default 512
             */
            width?: number;
            /**
             * Height
             * @description The height of the crop rectangle
             * @default 512
             */
            height?: number;
            /**
             * type
             * @default img_crop
             * @constant
             */
            type: "img_crop";
        };
        /**
         * ImageDTO
         * @description Deserialized image record, enriched for the frontend.
         */
        ImageDTO: {
            /**
             * Image Name
             * @description The unique name of the image.
             */
            image_name: string;
            /**
             * Image Url
             * @description The URL of the image.
             */
            image_url: string;
            /**
             * Thumbnail Url
             * @description The URL of the image's thumbnail.
             */
            thumbnail_url: string;
            /** @description The type of the image. */
            image_origin: components["schemas"]["ResourceOrigin"];
            /** @description The category of the image. */
            image_category: components["schemas"]["ImageCategory"];
            /**
             * Width
             * @description The width of the image in px.
             */
            width: number;
            /**
             * Height
             * @description The height of the image in px.
             */
            height: number;
            /**
             * Created At
             * @description The created timestamp of the image.
             */
            created_at: string;
            /**
             * Updated At
             * @description The updated timestamp of the image.
             */
            updated_at: string;
            /**
             * Deleted At
             * @description The deleted timestamp of the image.
             */
            deleted_at?: string | null;
            /**
             * Is Intermediate
             * @description Whether this is an intermediate image.
             */
            is_intermediate: boolean;
            /**
             * Session Id
             * @description The session ID that generated this image, if it is a generated image.
             */
            session_id?: string | null;
            /**
             * Node Id
             * @description The node ID that generated this image, if it is a generated image.
             */
            node_id?: string | null;
            /**
             * Starred
             * @description Whether this image is starred.
             */
            starred: boolean;
            /**
             * Has Workflow
             * @description Whether this image has a workflow.
             */
            has_workflow: boolean;
            /**
             * Board Id
             * @description The id of the board the image belongs to, if one exists.
             */
            board_id?: string | null;
        };
        /**
         * ImageField
         * @description An image primitive field
         */
        ImageField: {
            /**
             * Image Name
             * @description The name of the image
             */
            image_name: string;
        };
        /**
         * Image Generator
         * @description Generated a collection of images for use in a batched generation
         */
        ImageGenerator: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Generator Type
             * @description The image generator.
             */
            generator: components["schemas"]["ImageGeneratorField"];
            /**
             * type
             * @default image_generator
             * @constant
             */
            type: "image_generator";
        };
        /** ImageGeneratorField */
        ImageGeneratorField: Record<string, never>;
        /**
         * ImageGeneratorOutput
         * @description Base class for nodes that output a collection of boards
         */
        ImageGeneratorOutput: {
            /**
             * Images
             * @description The generated images
             */
            images: components["schemas"]["ImageField"][];
            /**
             * type
             * @default image_generator_output
             * @constant
             */
            type: "image_generator_output";
        };
        /**
         * Adjust Image Hue
         * @description Adjusts the Hue of an image.
         */
        ImageHueAdjustmentInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to adjust
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Hue
             * @description The degrees by which to rotate the hue, 0-360
             * @default 0
             */
            hue?: number;
            /**
             * type
             * @default img_hue_adjust
             * @constant
             */
            type: "img_hue_adjust";
        };
        /**
         * Inverse Lerp Image
         * @description Inverse linear interpolation of all pixels of an image
         */
        ImageInverseLerpInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to lerp
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Min
             * @description The minimum input value
             * @default 0
             */
            min?: number;
            /**
             * Max
             * @description The maximum input value
             * @default 255
             */
            max?: number;
            /**
             * type
             * @default img_ilerp
             * @constant
             */
            type: "img_ilerp";
        };
        /**
         * Image Primitive
         * @description An image primitive value
         */
        ImageInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to load
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default image
             * @constant
             */
            type: "image";
        };
        /**
         * Lerp Image
         * @description Linear interpolation of all pixels of an image
         */
        ImageLerpInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to lerp
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Min
             * @description The minimum output value
             * @default 0
             */
            min?: number;
            /**
             * Max
             * @description The maximum output value
             * @default 255
             */
            max?: number;
            /**
             * type
             * @default img_lerp
             * @constant
             */
            type: "img_lerp";
        };
        /**
         * Image Mask to Tensor
         * @description Convert a mask image to a tensor. Converts the image to grayscale and uses thresholding at the specified value.
         */
        ImageMaskToTensorInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The mask image to convert.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Cutoff
             * @description Cutoff (<)
             * @default 128
             */
            cutoff?: number;
            /**
             * Invert
             * @description Whether to invert the mask.
             * @default false
             */
            invert?: boolean;
            /**
             * type
             * @default image_mask_to_tensor
             * @constant
             */
            type: "image_mask_to_tensor";
        };
        /**
         * Multiply Images
         * @description Multiplies two images together using `PIL.ImageChops.multiply()`.
         */
        ImageMultiplyInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The first image to multiply
             * @default null
             */
            image1?: components["schemas"]["ImageField"];
            /**
             * @description The second image to multiply
             * @default null
             */
            image2?: components["schemas"]["ImageField"];
            /**
             * type
             * @default img_mul
             * @constant
             */
            type: "img_mul";
        };
        /**
         * Blur NSFW Image
         * @description Add blur to NSFW-flagged images
         */
        ImageNSFWBlurInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to check
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default img_nsfw
             * @constant
             */
            type: "img_nsfw";
        };
        /**
         * Add Image Noise
         * @description Add noise to an image
         */
        ImageNoiseInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to add noise to
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Seed
             * @description Seed for random number generation
             * @default 0
             */
            seed?: number;
            /**
             * Noise Type
             * @description The type of noise to add
             * @default gaussian
             * @enum {string}
             */
            noise_type?: "gaussian" | "salt_and_pepper";
            /**
             * Amount
             * @description The amount of noise to add
             * @default 0.1
             */
            amount?: number;
            /**
             * Noise Color
             * @description Whether to add colored noise
             * @default true
             */
            noise_color?: boolean;
            /**
             * Size
             * @description The size of the noise points
             * @default 1
             */
            size?: number;
            /**
             * type
             * @default img_noise
             * @constant
             */
            type: "img_noise";
        };
        /**
         * ImageOutput
         * @description Base class for nodes that output a single image
         */
        ImageOutput: {
            /** @description The output image */
            image: components["schemas"]["ImageField"];
            /**
             * Width
             * @description The width of the image in pixels
             */
            width: number;
            /**
             * Height
             * @description The height of the image in pixels
             */
            height: number;
            /**
             * type
             * @default image_output
             * @constant
             */
            type: "image_output";
        };
        /** ImagePanelCoordinateOutput */
        ImagePanelCoordinateOutput: {
            /**
             * X Left
             * @description The left x-coordinate of the panel.
             */
            x_left: number;
            /**
             * Y Top
             * @description The top y-coordinate of the panel.
             */
            y_top: number;
            /**
             * Width
             * @description The width of the panel.
             */
            width: number;
            /**
             * Height
             * @description The height of the panel.
             */
            height: number;
            /**
             * type
             * @default image_panel_coordinate_output
             * @constant
             */
            type: "image_panel_coordinate_output";
        };
        /**
         * Image Panel Layout
         * @description Get the coordinates of a single panel in a grid. (If the full image shape cannot be divided evenly into panels,
         *     then the grid may not cover the entire image.)
         */
        ImagePanelLayoutInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Width
             * @description The width of the entire grid.
             * @default null
             */
            width?: number;
            /**
             * Height
             * @description The height of the entire grid.
             * @default null
             */
            height?: number;
            /**
             * Num Cols
             * @description The number of columns in the grid.
             * @default 1
             */
            num_cols?: number;
            /**
             * Num Rows
             * @description The number of rows in the grid.
             * @default 1
             */
            num_rows?: number;
            /**
             * Panel Col Idx
             * @description The column index of the panel to be processed.
             * @default 0
             */
            panel_col_idx?: number;
            /**
             * Panel Row Idx
             * @description The row index of the panel to be processed.
             * @default 0
             */
            panel_row_idx?: number;
            /**
             * type
             * @default image_panel_layout
             * @constant
             */
            type: "image_panel_layout";
        };
        /**
         * Paste Image
         * @description Pastes an image into another image.
         */
        ImagePasteInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The base image
             * @default null
             */
            base_image?: components["schemas"]["ImageField"];
            /**
             * @description The image to paste
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The mask to use when pasting
             * @default null
             */
            mask?: components["schemas"]["ImageField"] | null;
            /**
             * X
             * @description The left x coordinate at which to paste the image
             * @default 0
             */
            x?: number;
            /**
             * Y
             * @description The top y coordinate at which to paste the image
             * @default 0
             */
            y?: number;
            /**
             * Crop
             * @description Crop to base image dimensions
             * @default false
             */
            crop?: boolean;
            /**
             * type
             * @default img_paste
             * @constant
             */
            type: "img_paste";
        };
        /**
         * ImageRecordChanges
         * @description A set of changes to apply to an image record.
         *
         *     Only limited changes are valid:
         *       - `image_category`: change the category of an image
         *       - `session_id`: change the session associated with an image
         *       - `is_intermediate`: change the image's `is_intermediate` flag
         *       - `starred`: change whether the image is starred
         */
        ImageRecordChanges: {
            /** @description The image's new category. */
            image_category?: components["schemas"]["ImageCategory"] | null;
            /**
             * Session Id
             * @description The image's new session ID.
             */
            session_id?: string | null;
            /**
             * Is Intermediate
             * @description The image's new `is_intermediate` flag.
             */
            is_intermediate?: boolean | null;
            /**
             * Starred
             * @description The image's new `starred` state
             */
            starred?: boolean | null;
        } & {
            [key: string]: unknown;
        };
        /**
         * Resize Image
         * @description Resizes an image to specific dimensions
         */
        ImageResizeInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to resize
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Width
             * @description The width to resize to (px)
             * @default 512
             */
            width?: number;
            /**
             * Height
             * @description The height to resize to (px)
             * @default 512
             */
            height?: number;
            /**
             * Resample Mode
             * @description The resampling mode
             * @default bicubic
             * @enum {string}
             */
            resample_mode?: "nearest" | "box" | "bilinear" | "hamming" | "bicubic" | "lanczos";
            /**
             * type
             * @default img_resize
             * @constant
             */
            type: "img_resize";
        };
        /**
         * Scale Image
         * @description Scales an image by a factor
         */
        ImageScaleInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to scale
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Scale Factor
             * @description The factor by which to scale the image
             * @default 2
             */
            scale_factor?: number;
            /**
             * Resample Mode
             * @description The resampling mode
             * @default bicubic
             * @enum {string}
             */
            resample_mode?: "nearest" | "box" | "bilinear" | "hamming" | "bicubic" | "lanczos";
            /**
             * type
             * @default img_scale
             * @constant
             */
            type: "img_scale";
        };
        /**
         * Image to Latents - SD1.5, SDXL
         * @description Encodes an image into latents.
         */
        ImageToLatentsInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to encode
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * Tiled
             * @description Processing using overlapping tiles (reduce memory consumption)
             * @default false
             */
            tiled?: boolean;
            /**
             * Tile Size
             * @description The tile size for VAE tiling in pixels (image space). If set to 0, the default tile size for the model will be used. Larger tile sizes generally produce better results at the cost of higher memory usage.
             * @default 0
             */
            tile_size?: number;
            /**
             * Fp32
             * @description Whether or not to use full float32 precision
             * @default false
             */
            fp32?: boolean;
            /**
             * type
             * @default i2l
             * @constant
             */
            type: "i2l";
        };
        /** ImageUploadEntry */
        ImageUploadEntry: {
            /** @description The image DTO */
            image_dto: components["schemas"]["ImageDTO"];
            /**
             * Presigned Url
             * @description The URL to get the presigned URL for the image upload
             */
            presigned_url: string;
        };
        /**
         * ImageUrlsDTO
         * @description The URLs for an image and its thumbnail.
         */
        ImageUrlsDTO: {
            /**
             * Image Name
             * @description The unique name of the image.
             */
            image_name: string;
            /**
             * Image Url
             * @description The URL of the image.
             */
            image_url: string;
            /**
             * Thumbnail Url
             * @description The URL of the image's thumbnail.
             */
            thumbnail_url: string;
        };
        /**
         * Add Invisible Watermark
         * @description Add an invisible watermark to an image
         */
        ImageWatermarkInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to check
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Text
             * @description Watermark text
             * @default InvokeAI
             */
            text?: string;
            /**
             * type
             * @default img_watermark
             * @constant
             */
            type: "img_watermark";
        };
        /** ImagesDownloaded */
        ImagesDownloaded: {
            /**
             * Response
             * @description The message to display to the user when images begin downloading
             */
            response?: string | null;
            /**
             * Bulk Download Item Name
             * @description The name of the bulk download item for which events will be emitted
             */
            bulk_download_item_name?: string | null;
        };
        /** ImagesUpdatedFromListResult */
        ImagesUpdatedFromListResult: {
            /**
             * Updated Image Names
             * @description The image names that were updated
             */
            updated_image_names: string[];
        };
        /**
         * Solid Color Infill
         * @description Infills transparent areas of an image with a solid color
         */
        InfillColorInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The color to use to infill
             * @default {
             *       "r": 127,
             *       "g": 127,
             *       "b": 127,
             *       "a": 255
             *     }
             */
            color?: components["schemas"]["ColorField"];
            /**
             * type
             * @default infill_rgba
             * @constant
             */
            type: "infill_rgba";
        };
        /**
         * PatchMatch Infill
         * @description Infills transparent areas of an image using the PatchMatch algorithm
         */
        InfillPatchMatchInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Downscale
             * @description Run patchmatch on downscaled image to speedup infill
             * @default 2
             */
            downscale?: number;
            /**
             * Resample Mode
             * @description The resampling mode
             * @default bicubic
             * @enum {string}
             */
            resample_mode?: "nearest" | "box" | "bilinear" | "hamming" | "bicubic" | "lanczos";
            /**
             * type
             * @default infill_patchmatch
             * @constant
             */
            type: "infill_patchmatch";
        };
        /**
         * Tile Infill
         * @description Infills transparent areas of an image with tiles of the image
         */
        InfillTileInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Tile Size
             * @description The tile size (px)
             * @default 32
             */
            tile_size?: number;
            /**
             * Seed
             * @description The seed to use for tile generation (omit for random)
             * @default 0
             */
            seed?: number;
            /**
             * type
             * @default infill_tile
             * @constant
             */
            type: "infill_tile";
        };
        /**
         * Input
         * @description The type of input a field accepts.
         *     - `Input.Direct`: The field must have its value provided directly, when the invocation and field       are instantiated.
         *     - `Input.Connection`: The field must have its value provided by a connection.
         *     - `Input.Any`: The field may have its value provided either directly or by a connection.
         * @enum {string}
         */
        Input: "connection" | "direct" | "any";
        /**
         * InputFieldJSONSchemaExtra
         * @description Extra attributes to be added to input fields and their OpenAPI schema. Used during graph execution,
         *     and by the workflow editor during schema parsing and UI rendering.
         */
        InputFieldJSONSchemaExtra: {
            input: components["schemas"]["Input"];
            /** Orig Required */
            orig_required: boolean;
            field_kind: components["schemas"]["FieldKind"];
            /**
             * Default
             * @default null
             */
            default: unknown | null;
            /**
             * Orig Default
             * @default null
             */
            orig_default: unknown | null;
            /**
             * Ui Hidden
             * @default false
             */
            ui_hidden: boolean;
            /** @default null */
            ui_type: components["schemas"]["UIType"] | null;
            /** @default null */
            ui_component: components["schemas"]["UIComponent"] | null;
            /**
             * Ui Order
             * @default null
             */
            ui_order: number | null;
            /**
             * Ui Choice Labels
             * @default null
             */
            ui_choice_labels: {
                [key: string]: string;
            } | null;
        };
        /**
         * InstallStatus
         * @description State of an install job running in the background.
         * @enum {string}
         */
        InstallStatus: "waiting" | "downloading" | "downloads_done" | "running" | "completed" | "error" | "cancelled";
        /**
         * Integer Batch
         * @description Create a batched generation, where the workflow is executed once for each integer in the batch.
         */
        IntegerBatchInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Batch Group
             * @description The ID of this batch node's group. If provided, all batch nodes in with the same ID will be 'zipped' before execution, and all nodes' collections must be of the same size.
             * @default None
             * @enum {string}
             */
            batch_group_id?: "None" | "Group 1" | "Group 2" | "Group 3" | "Group 4" | "Group 5";
            /**
             * Integers
             * @description The integers to batch over
             * @default []
             */
            integers?: number[];
            /**
             * type
             * @default integer_batch
             * @constant
             */
            type: "integer_batch";
        };
        /**
         * Integer Collection Primitive
         * @description A collection of integer primitive values
         */
        IntegerCollectionInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The collection of integer values
             * @default []
             */
            collection?: number[];
            /**
             * type
             * @default integer_collection
             * @constant
             */
            type: "integer_collection";
        };
        /**
         * IntegerCollectionOutput
         * @description Base class for nodes that output a collection of integers
         */
        IntegerCollectionOutput: {
            /**
             * Collection
             * @description The int collection
             */
            collection: number[];
            /**
             * type
             * @default integer_collection_output
             * @constant
             */
            type: "integer_collection_output";
        };
        /**
         * Integer Generator
         * @description Generated a range of integers for use in a batched generation
         */
        IntegerGenerator: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Generator Type
             * @description The integer generator.
             */
            generator: components["schemas"]["IntegerGeneratorField"];
            /**
             * type
             * @default integer_generator
             * @constant
             */
            type: "integer_generator";
        };
        /** IntegerGeneratorField */
        IntegerGeneratorField: Record<string, never>;
        /** IntegerGeneratorOutput */
        IntegerGeneratorOutput: {
            /**
             * Integers
             * @description The generated integers
             */
            integers: number[];
            /**
             * type
             * @default integer_generator_output
             * @constant
             */
            type: "integer_generator_output";
        };
        /**
         * Integer Primitive
         * @description An integer primitive value
         */
        IntegerInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Value
             * @description The integer value
             * @default 0
             */
            value?: number;
            /**
             * type
             * @default integer
             * @constant
             */
            type: "integer";
        };
        /**
         * Integer Math
         * @description Performs integer math.
         */
        IntegerMathInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Operation
             * @description The operation to perform
             * @default ADD
             * @enum {string}
             */
            operation?: "ADD" | "SUB" | "MUL" | "DIV" | "EXP" | "MOD" | "ABS" | "MIN" | "MAX";
            /**
             * A
             * @description The first number
             * @default 1
             */
            a?: number;
            /**
             * B
             * @description The second number
             * @default 1
             */
            b?: number;
            /**
             * type
             * @default integer_math
             * @constant
             */
            type: "integer_math";
        };
        /**
         * IntegerOutput
         * @description Base class for nodes that output a single integer
         */
        IntegerOutput: {
            /**
             * Value
             * @description The output integer
             */
            value: number;
            /**
             * type
             * @default integer_output
             * @constant
             */
            type: "integer_output";
        };
        /**
         * Invert Tensor Mask
         * @description Inverts a tensor mask.
         */
        InvertTensorMaskInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The tensor mask to convert.
             * @default null
             */
            mask?: components["schemas"]["TensorField"];
            /**
             * type
             * @default invert_tensor_mask
             * @constant
             */
            type: "invert_tensor_mask";
        };
        /** InvocationCacheStatus */
        InvocationCacheStatus: {
            /**
             * Size
             * @description The current size of the invocation cache
             */
            size: number;
            /**
             * Hits
             * @description The number of cache hits
             */
            hits: number;
            /**
             * Misses
             * @description The number of cache misses
             */
            misses: number;
            /**
             * Enabled
             * @description Whether the invocation cache is enabled
             */
            enabled: boolean;
            /**
             * Max Size
             * @description The maximum size of the invocation cache
             */
            max_size: number;
        };
        /**
         * InvocationCompleteEvent
         * @description Event model for invocation_complete
         */
        InvocationCompleteEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Item Id
             * @description The ID of the queue item
             */
            item_id: number;
            /**
             * Batch Id
             * @description The ID of the queue batch
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of the queue item
             * @default null
             */
            origin: string | null;
            /**
             * Destination
             * @description The destination of the queue item
             * @default null
             */
            destination: string | null;
            /**
             * Session Id
             * @description The ID of the session (aka graph execution state)
             */
            session_id: string;
            /**
             * Invocation
             * @description The ID of the invocation
             */
            invocation: components["schemas"]["AddInvocation"] | components["schemas"]["AlphaMaskToTensorInvocation"] | components["schemas"]["ApplyMaskTensorToImageInvocation"] | components["schemas"]["ApplyMaskToImageInvocation"] | components["schemas"]["BlankImageInvocation"] | components["schemas"]["BlendLatentsInvocation"] | components["schemas"]["BooleanCollectionInvocation"] | components["schemas"]["BooleanInvocation"] | components["schemas"]["BoundingBoxInvocation"] | components["schemas"]["CLIPSkipInvocation"] | components["schemas"]["CV2InfillInvocation"] | components["schemas"]["CalculateImageTilesEvenSplitInvocation"] | components["schemas"]["CalculateImageTilesInvocation"] | components["schemas"]["CalculateImageTilesMinimumOverlapInvocation"] | components["schemas"]["CannyEdgeDetectionInvocation"] | components["schemas"]["CanvasPasteBackInvocation"] | components["schemas"]["CanvasV2MaskAndCropInvocation"] | components["schemas"]["CenterPadCropInvocation"] | components["schemas"]["CogView4DenoiseInvocation"] | components["schemas"]["CogView4ImageToLatentsInvocation"] | components["schemas"]["CogView4LatentsToImageInvocation"] | components["schemas"]["CogView4ModelLoaderInvocation"] | components["schemas"]["CogView4TextEncoderInvocation"] | components["schemas"]["CollectInvocation"] | components["schemas"]["ColorCorrectInvocation"] | components["schemas"]["ColorInvocation"] | components["schemas"]["ColorMapInvocation"] | components["schemas"]["CompelInvocation"] | components["schemas"]["ConditioningCollectionInvocation"] | components["schemas"]["ConditioningInvocation"] | components["schemas"]["ContentShuffleInvocation"] | components["schemas"]["ControlNetInvocation"] | components["schemas"]["CoreMetadataInvocation"] | components["schemas"]["CreateDenoiseMaskInvocation"] | components["schemas"]["CreateGradientMaskInvocation"] | components["schemas"]["CropImageToBoundingBoxInvocation"] | components["schemas"]["CropLatentsCoreInvocation"] | components["schemas"]["CvInpaintInvocation"] | components["schemas"]["DWOpenposeDetectionInvocation"] | components["schemas"]["DenoiseLatentsInvocation"] | components["schemas"]["DenoiseLatentsMetaInvocation"] | components["schemas"]["DepthAnythingDepthEstimationInvocation"] | components["schemas"]["DivideInvocation"] | components["schemas"]["DynamicPromptInvocation"] | components["schemas"]["ESRGANInvocation"] | components["schemas"]["ExpandMaskWithFadeInvocation"] | components["schemas"]["FLUXLoRACollectionLoader"] | components["schemas"]["FaceIdentifierInvocation"] | components["schemas"]["FaceMaskInvocation"] | components["schemas"]["FaceOffInvocation"] | components["schemas"]["FloatBatchInvocation"] | components["schemas"]["FloatCollectionInvocation"] | components["schemas"]["FloatGenerator"] | components["schemas"]["FloatInvocation"] | components["schemas"]["FloatLinearRangeInvocation"] | components["schemas"]["FloatMathInvocation"] | components["schemas"]["FloatToIntegerInvocation"] | components["schemas"]["FluxControlLoRALoaderInvocation"] | components["schemas"]["FluxControlNetInvocation"] | components["schemas"]["FluxDenoiseInvocation"] | components["schemas"]["FluxDenoiseLatentsMetaInvocation"] | components["schemas"]["FluxFillInvocation"] | components["schemas"]["FluxIPAdapterInvocation"] | components["schemas"]["FluxLoRALoaderInvocation"] | components["schemas"]["FluxModelLoaderInvocation"] | components["schemas"]["FluxReduxInvocation"] | components["schemas"]["FluxTextEncoderInvocation"] | components["schemas"]["FluxVaeDecodeInvocation"] | components["schemas"]["FluxVaeEncodeInvocation"] | components["schemas"]["FreeUInvocation"] | components["schemas"]["GetMaskBoundingBoxInvocation"] | components["schemas"]["GroundingDinoInvocation"] | components["schemas"]["HEDEdgeDetectionInvocation"] | components["schemas"]["HeuristicResizeInvocation"] | components["schemas"]["IPAdapterInvocation"] | components["schemas"]["IdealSizeInvocation"] | components["schemas"]["ImageBatchInvocation"] | components["schemas"]["ImageBlurInvocation"] | components["schemas"]["ImageChannelInvocation"] | components["schemas"]["ImageChannelMultiplyInvocation"] | components["schemas"]["ImageChannelOffsetInvocation"] | components["schemas"]["ImageCollectionInvocation"] | components["schemas"]["ImageConvertInvocation"] | components["schemas"]["ImageCropInvocation"] | components["schemas"]["ImageGenerator"] | components["schemas"]["ImageHueAdjustmentInvocation"] | components["schemas"]["ImageInverseLerpInvocation"] | components["schemas"]["ImageInvocation"] | components["schemas"]["ImageLerpInvocation"] | components["schemas"]["ImageMaskToTensorInvocation"] | components["schemas"]["ImageMultiplyInvocation"] | components["schemas"]["ImageNSFWBlurInvocation"] | components["schemas"]["ImageNoiseInvocation"] | components["schemas"]["ImagePanelLayoutInvocation"] | components["schemas"]["ImagePasteInvocation"] | components["schemas"]["ImageResizeInvocation"] | components["schemas"]["ImageScaleInvocation"] | components["schemas"]["ImageToLatentsInvocation"] | components["schemas"]["ImageWatermarkInvocation"] | components["schemas"]["InfillColorInvocation"] | components["schemas"]["InfillPatchMatchInvocation"] | components["schemas"]["InfillTileInvocation"] | components["schemas"]["IntegerBatchInvocation"] | components["schemas"]["IntegerCollectionInvocation"] | components["schemas"]["IntegerGenerator"] | components["schemas"]["IntegerInvocation"] | components["schemas"]["IntegerMathInvocation"] | components["schemas"]["InvertTensorMaskInvocation"] | components["schemas"]["InvokeAdjustImageHuePlusInvocation"] | components["schemas"]["InvokeEquivalentAchromaticLightnessInvocation"] | components["schemas"]["InvokeImageBlendInvocation"] | components["schemas"]["InvokeImageCompositorInvocation"] | components["schemas"]["InvokeImageDilateOrErodeInvocation"] | components["schemas"]["InvokeImageEnhanceInvocation"] | components["schemas"]["InvokeImageValueThresholdsInvocation"] | components["schemas"]["IterateInvocation"] | components["schemas"]["LaMaInfillInvocation"] | components["schemas"]["LatentsCollectionInvocation"] | components["schemas"]["LatentsInvocation"] | components["schemas"]["LatentsToImageInvocation"] | components["schemas"]["LineartAnimeEdgeDetectionInvocation"] | components["schemas"]["LineartEdgeDetectionInvocation"] | components["schemas"]["LlavaOnevisionVllmInvocation"] | components["schemas"]["LoRACollectionLoader"] | components["schemas"]["LoRALoaderInvocation"] | components["schemas"]["LoRASelectorInvocation"] | components["schemas"]["MLSDDetectionInvocation"] | components["schemas"]["MainModelLoaderInvocation"] | components["schemas"]["MaskCombineInvocation"] | components["schemas"]["MaskEdgeInvocation"] | components["schemas"]["MaskFromAlphaInvocation"] | components["schemas"]["MaskFromIDInvocation"] | components["schemas"]["MaskTensorToImageInvocation"] | components["schemas"]["MediaPipeFaceDetectionInvocation"] | components["schemas"]["MergeMetadataInvocation"] | components["schemas"]["MergeTilesToImageInvocation"] | components["schemas"]["MetadataFieldExtractorInvocation"] | components["schemas"]["MetadataFromImageInvocation"] | components["schemas"]["MetadataInvocation"] | components["schemas"]["MetadataItemInvocation"] | components["schemas"]["MetadataItemLinkedInvocation"] | components["schemas"]["MetadataToBoolInvocation"] | components["schemas"]["MetadataToControlnetsInvocation"] | components["schemas"]["MetadataToFloatInvocation"] | components["schemas"]["MetadataToIPAdaptersInvocation"] | components["schemas"]["MetadataToIntegerInvocation"] | components["schemas"]["MetadataToLorasCollectionInvocation"] | components["schemas"]["MetadataToLorasInvocation"] | components["schemas"]["MetadataToModelInvocation"] | components["schemas"]["MetadataToSDXLLorasInvocation"] | components["schemas"]["MetadataToSDXLModelInvocation"] | components["schemas"]["MetadataToSchedulerInvocation"] | components["schemas"]["MetadataToStringInvocation"] | components["schemas"]["MetadataToT2IAdaptersInvocation"] | components["schemas"]["MetadataToVAEInvocation"] | components["schemas"]["ModelIdentifierInvocation"] | components["schemas"]["MultiplyInvocation"] | components["schemas"]["NoiseInvocation"] | components["schemas"]["NormalMapInvocation"] | components["schemas"]["PairTileImageInvocation"] | components["schemas"]["PasteImageIntoBoundingBoxInvocation"] | components["schemas"]["PiDiNetEdgeDetectionInvocation"] | components["schemas"]["PromptsFromFileInvocation"] | components["schemas"]["RandomFloatInvocation"] | components["schemas"]["RandomIntInvocation"] | components["schemas"]["RandomRangeInvocation"] | components["schemas"]["RangeInvocation"] | components["schemas"]["RangeOfSizeInvocation"] | components["schemas"]["RectangleMaskInvocation"] | components["schemas"]["ResizeLatentsInvocation"] | components["schemas"]["RoundInvocation"] | components["schemas"]["SD3DenoiseInvocation"] | components["schemas"]["SD3ImageToLatentsInvocation"] | components["schemas"]["SD3LatentsToImageInvocation"] | components["schemas"]["SDXLCompelPromptInvocation"] | components["schemas"]["SDXLLoRACollectionLoader"] | components["schemas"]["SDXLLoRALoaderInvocation"] | components["schemas"]["SDXLModelLoaderInvocation"] | components["schemas"]["SDXLRefinerCompelPromptInvocation"] | components["schemas"]["SDXLRefinerModelLoaderInvocation"] | components["schemas"]["SaveImageInvocation"] | components["schemas"]["ScaleLatentsInvocation"] | components["schemas"]["SchedulerInvocation"] | components["schemas"]["Sd3ModelLoaderInvocation"] | components["schemas"]["Sd3TextEncoderInvocation"] | components["schemas"]["SeamlessModeInvocation"] | components["schemas"]["SegmentAnythingInvocation"] | components["schemas"]["ShowImageInvocation"] | components["schemas"]["SpandrelImageToImageAutoscaleInvocation"] | components["schemas"]["SpandrelImageToImageInvocation"] | components["schemas"]["StringBatchInvocation"] | components["schemas"]["StringCollectionInvocation"] | components["schemas"]["StringGenerator"] | components["schemas"]["StringInvocation"] | components["schemas"]["StringJoinInvocation"] | components["schemas"]["StringJoinThreeInvocation"] | components["schemas"]["StringReplaceInvocation"] | components["schemas"]["StringSplitInvocation"] | components["schemas"]["StringSplitNegInvocation"] | components["schemas"]["SubtractInvocation"] | components["schemas"]["T2IAdapterInvocation"] | components["schemas"]["TileToPropertiesInvocation"] | components["schemas"]["TiledMultiDiffusionDenoiseLatents"] | components["schemas"]["UnsharpMaskInvocation"] | components["schemas"]["VAELoaderInvocation"];
            /**
             * Invocation Source Id
             * @description The ID of the prepared invocation's source node
             */
            invocation_source_id: string;
            /**
             * Result
             * @description The result of the invocation
             */
            result: components["schemas"]["BooleanCollectionOutput"] | components["schemas"]["BooleanOutput"] | components["schemas"]["BoundingBoxCollectionOutput"] | components["schemas"]["BoundingBoxOutput"] | components["schemas"]["CLIPOutput"] | components["schemas"]["CLIPSkipInvocationOutput"] | components["schemas"]["CalculateImageTilesOutput"] | components["schemas"]["CogView4ConditioningOutput"] | components["schemas"]["CogView4ModelLoaderOutput"] | components["schemas"]["CollectInvocationOutput"] | components["schemas"]["ColorCollectionOutput"] | components["schemas"]["ColorOutput"] | components["schemas"]["ConditioningCollectionOutput"] | components["schemas"]["ConditioningOutput"] | components["schemas"]["ControlOutput"] | components["schemas"]["DenoiseMaskOutput"] | components["schemas"]["FaceMaskOutput"] | components["schemas"]["FaceOffOutput"] | components["schemas"]["FloatCollectionOutput"] | components["schemas"]["FloatGeneratorOutput"] | components["schemas"]["FloatOutput"] | components["schemas"]["FluxConditioningOutput"] | components["schemas"]["FluxControlLoRALoaderOutput"] | components["schemas"]["FluxControlNetOutput"] | components["schemas"]["FluxFillOutput"] | components["schemas"]["FluxLoRALoaderOutput"] | components["schemas"]["FluxModelLoaderOutput"] | components["schemas"]["FluxReduxOutput"] | components["schemas"]["GradientMaskOutput"] | components["schemas"]["IPAdapterOutput"] | components["schemas"]["IdealSizeOutput"] | components["schemas"]["ImageCollectionOutput"] | components["schemas"]["ImageGeneratorOutput"] | components["schemas"]["ImageOutput"] | components["schemas"]["ImagePanelCoordinateOutput"] | components["schemas"]["IntegerCollectionOutput"] | components["schemas"]["IntegerGeneratorOutput"] | components["schemas"]["IntegerOutput"] | components["schemas"]["IterateInvocationOutput"] | components["schemas"]["LatentsCollectionOutput"] | components["schemas"]["LatentsMetaOutput"] | components["schemas"]["LatentsOutput"] | components["schemas"]["LoRALoaderOutput"] | components["schemas"]["LoRASelectorOutput"] | components["schemas"]["MDControlListOutput"] | components["schemas"]["MDIPAdapterListOutput"] | components["schemas"]["MDT2IAdapterListOutput"] | components["schemas"]["MaskOutput"] | components["schemas"]["MetadataItemOutput"] | components["schemas"]["MetadataOutput"] | components["schemas"]["MetadataToLorasCollectionOutput"] | components["schemas"]["MetadataToModelOutput"] | components["schemas"]["MetadataToSDXLModelOutput"] | components["schemas"]["ModelIdentifierOutput"] | components["schemas"]["ModelLoaderOutput"] | components["schemas"]["NoiseOutput"] | components["schemas"]["PairTileImageOutput"] | components["schemas"]["SD3ConditioningOutput"] | components["schemas"]["SDXLLoRALoaderOutput"] | components["schemas"]["SDXLModelLoaderOutput"] | components["schemas"]["SDXLRefinerModelLoaderOutput"] | components["schemas"]["SchedulerOutput"] | components["schemas"]["Sd3ModelLoaderOutput"] | components["schemas"]["SeamlessModeOutput"] | components["schemas"]["String2Output"] | components["schemas"]["StringCollectionOutput"] | components["schemas"]["StringGeneratorOutput"] | components["schemas"]["StringOutput"] | components["schemas"]["StringPosNegOutput"] | components["schemas"]["T2IAdapterOutput"] | components["schemas"]["TileToPropertiesOutput"] | components["schemas"]["UNetOutput"] | components["schemas"]["VAEOutput"];
        };
        /**
         * InvocationErrorEvent
         * @description Event model for invocation_error
         */
        InvocationErrorEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Item Id
             * @description The ID of the queue item
             */
            item_id: number;
            /**
             * Batch Id
             * @description The ID of the queue batch
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of the queue item
             * @default null
             */
            origin: string | null;
            /**
             * Destination
             * @description The destination of the queue item
             * @default null
             */
            destination: string | null;
            /**
             * Session Id
             * @description The ID of the session (aka graph execution state)
             */
            session_id: string;
            /**
             * Invocation
             * @description The ID of the invocation
             */
            invocation: components["schemas"]["AddInvocation"] | components["schemas"]["AlphaMaskToTensorInvocation"] | components["schemas"]["ApplyMaskTensorToImageInvocation"] | components["schemas"]["ApplyMaskToImageInvocation"] | components["schemas"]["BlankImageInvocation"] | components["schemas"]["BlendLatentsInvocation"] | components["schemas"]["BooleanCollectionInvocation"] | components["schemas"]["BooleanInvocation"] | components["schemas"]["BoundingBoxInvocation"] | components["schemas"]["CLIPSkipInvocation"] | components["schemas"]["CV2InfillInvocation"] | components["schemas"]["CalculateImageTilesEvenSplitInvocation"] | components["schemas"]["CalculateImageTilesInvocation"] | components["schemas"]["CalculateImageTilesMinimumOverlapInvocation"] | components["schemas"]["CannyEdgeDetectionInvocation"] | components["schemas"]["CanvasPasteBackInvocation"] | components["schemas"]["CanvasV2MaskAndCropInvocation"] | components["schemas"]["CenterPadCropInvocation"] | components["schemas"]["CogView4DenoiseInvocation"] | components["schemas"]["CogView4ImageToLatentsInvocation"] | components["schemas"]["CogView4LatentsToImageInvocation"] | components["schemas"]["CogView4ModelLoaderInvocation"] | components["schemas"]["CogView4TextEncoderInvocation"] | components["schemas"]["CollectInvocation"] | components["schemas"]["ColorCorrectInvocation"] | components["schemas"]["ColorInvocation"] | components["schemas"]["ColorMapInvocation"] | components["schemas"]["CompelInvocation"] | components["schemas"]["ConditioningCollectionInvocation"] | components["schemas"]["ConditioningInvocation"] | components["schemas"]["ContentShuffleInvocation"] | components["schemas"]["ControlNetInvocation"] | components["schemas"]["CoreMetadataInvocation"] | components["schemas"]["CreateDenoiseMaskInvocation"] | components["schemas"]["CreateGradientMaskInvocation"] | components["schemas"]["CropImageToBoundingBoxInvocation"] | components["schemas"]["CropLatentsCoreInvocation"] | components["schemas"]["CvInpaintInvocation"] | components["schemas"]["DWOpenposeDetectionInvocation"] | components["schemas"]["DenoiseLatentsInvocation"] | components["schemas"]["DenoiseLatentsMetaInvocation"] | components["schemas"]["DepthAnythingDepthEstimationInvocation"] | components["schemas"]["DivideInvocation"] | components["schemas"]["DynamicPromptInvocation"] | components["schemas"]["ESRGANInvocation"] | components["schemas"]["ExpandMaskWithFadeInvocation"] | components["schemas"]["FLUXLoRACollectionLoader"] | components["schemas"]["FaceIdentifierInvocation"] | components["schemas"]["FaceMaskInvocation"] | components["schemas"]["FaceOffInvocation"] | components["schemas"]["FloatBatchInvocation"] | components["schemas"]["FloatCollectionInvocation"] | components["schemas"]["FloatGenerator"] | components["schemas"]["FloatInvocation"] | components["schemas"]["FloatLinearRangeInvocation"] | components["schemas"]["FloatMathInvocation"] | components["schemas"]["FloatToIntegerInvocation"] | components["schemas"]["FluxControlLoRALoaderInvocation"] | components["schemas"]["FluxControlNetInvocation"] | components["schemas"]["FluxDenoiseInvocation"] | components["schemas"]["FluxDenoiseLatentsMetaInvocation"] | components["schemas"]["FluxFillInvocation"] | components["schemas"]["FluxIPAdapterInvocation"] | components["schemas"]["FluxLoRALoaderInvocation"] | components["schemas"]["FluxModelLoaderInvocation"] | components["schemas"]["FluxReduxInvocation"] | components["schemas"]["FluxTextEncoderInvocation"] | components["schemas"]["FluxVaeDecodeInvocation"] | components["schemas"]["FluxVaeEncodeInvocation"] | components["schemas"]["FreeUInvocation"] | components["schemas"]["GetMaskBoundingBoxInvocation"] | components["schemas"]["GroundingDinoInvocation"] | components["schemas"]["HEDEdgeDetectionInvocation"] | components["schemas"]["HeuristicResizeInvocation"] | components["schemas"]["IPAdapterInvocation"] | components["schemas"]["IdealSizeInvocation"] | components["schemas"]["ImageBatchInvocation"] | components["schemas"]["ImageBlurInvocation"] | components["schemas"]["ImageChannelInvocation"] | components["schemas"]["ImageChannelMultiplyInvocation"] | components["schemas"]["ImageChannelOffsetInvocation"] | components["schemas"]["ImageCollectionInvocation"] | components["schemas"]["ImageConvertInvocation"] | components["schemas"]["ImageCropInvocation"] | components["schemas"]["ImageGenerator"] | components["schemas"]["ImageHueAdjustmentInvocation"] | components["schemas"]["ImageInverseLerpInvocation"] | components["schemas"]["ImageInvocation"] | components["schemas"]["ImageLerpInvocation"] | components["schemas"]["ImageMaskToTensorInvocation"] | components["schemas"]["ImageMultiplyInvocation"] | components["schemas"]["ImageNSFWBlurInvocation"] | components["schemas"]["ImageNoiseInvocation"] | components["schemas"]["ImagePanelLayoutInvocation"] | components["schemas"]["ImagePasteInvocation"] | components["schemas"]["ImageResizeInvocation"] | components["schemas"]["ImageScaleInvocation"] | components["schemas"]["ImageToLatentsInvocation"] | components["schemas"]["ImageWatermarkInvocation"] | components["schemas"]["InfillColorInvocation"] | components["schemas"]["InfillPatchMatchInvocation"] | components["schemas"]["InfillTileInvocation"] | components["schemas"]["IntegerBatchInvocation"] | components["schemas"]["IntegerCollectionInvocation"] | components["schemas"]["IntegerGenerator"] | components["schemas"]["IntegerInvocation"] | components["schemas"]["IntegerMathInvocation"] | components["schemas"]["InvertTensorMaskInvocation"] | components["schemas"]["InvokeAdjustImageHuePlusInvocation"] | components["schemas"]["InvokeEquivalentAchromaticLightnessInvocation"] | components["schemas"]["InvokeImageBlendInvocation"] | components["schemas"]["InvokeImageCompositorInvocation"] | components["schemas"]["InvokeImageDilateOrErodeInvocation"] | components["schemas"]["InvokeImageEnhanceInvocation"] | components["schemas"]["InvokeImageValueThresholdsInvocation"] | components["schemas"]["IterateInvocation"] | components["schemas"]["LaMaInfillInvocation"] | components["schemas"]["LatentsCollectionInvocation"] | components["schemas"]["LatentsInvocation"] | components["schemas"]["LatentsToImageInvocation"] | components["schemas"]["LineartAnimeEdgeDetectionInvocation"] | components["schemas"]["LineartEdgeDetectionInvocation"] | components["schemas"]["LlavaOnevisionVllmInvocation"] | components["schemas"]["LoRACollectionLoader"] | components["schemas"]["LoRALoaderInvocation"] | components["schemas"]["LoRASelectorInvocation"] | components["schemas"]["MLSDDetectionInvocation"] | components["schemas"]["MainModelLoaderInvocation"] | components["schemas"]["MaskCombineInvocation"] | components["schemas"]["MaskEdgeInvocation"] | components["schemas"]["MaskFromAlphaInvocation"] | components["schemas"]["MaskFromIDInvocation"] | components["schemas"]["MaskTensorToImageInvocation"] | components["schemas"]["MediaPipeFaceDetectionInvocation"] | components["schemas"]["MergeMetadataInvocation"] | components["schemas"]["MergeTilesToImageInvocation"] | components["schemas"]["MetadataFieldExtractorInvocation"] | components["schemas"]["MetadataFromImageInvocation"] | components["schemas"]["MetadataInvocation"] | components["schemas"]["MetadataItemInvocation"] | components["schemas"]["MetadataItemLinkedInvocation"] | components["schemas"]["MetadataToBoolInvocation"] | components["schemas"]["MetadataToControlnetsInvocation"] | components["schemas"]["MetadataToFloatInvocation"] | components["schemas"]["MetadataToIPAdaptersInvocation"] | components["schemas"]["MetadataToIntegerInvocation"] | components["schemas"]["MetadataToLorasCollectionInvocation"] | components["schemas"]["MetadataToLorasInvocation"] | components["schemas"]["MetadataToModelInvocation"] | components["schemas"]["MetadataToSDXLLorasInvocation"] | components["schemas"]["MetadataToSDXLModelInvocation"] | components["schemas"]["MetadataToSchedulerInvocation"] | components["schemas"]["MetadataToStringInvocation"] | components["schemas"]["MetadataToT2IAdaptersInvocation"] | components["schemas"]["MetadataToVAEInvocation"] | components["schemas"]["ModelIdentifierInvocation"] | components["schemas"]["MultiplyInvocation"] | components["schemas"]["NoiseInvocation"] | components["schemas"]["NormalMapInvocation"] | components["schemas"]["PairTileImageInvocation"] | components["schemas"]["PasteImageIntoBoundingBoxInvocation"] | components["schemas"]["PiDiNetEdgeDetectionInvocation"] | components["schemas"]["PromptsFromFileInvocation"] | components["schemas"]["RandomFloatInvocation"] | components["schemas"]["RandomIntInvocation"] | components["schemas"]["RandomRangeInvocation"] | components["schemas"]["RangeInvocation"] | components["schemas"]["RangeOfSizeInvocation"] | components["schemas"]["RectangleMaskInvocation"] | components["schemas"]["ResizeLatentsInvocation"] | components["schemas"]["RoundInvocation"] | components["schemas"]["SD3DenoiseInvocation"] | components["schemas"]["SD3ImageToLatentsInvocation"] | components["schemas"]["SD3LatentsToImageInvocation"] | components["schemas"]["SDXLCompelPromptInvocation"] | components["schemas"]["SDXLLoRACollectionLoader"] | components["schemas"]["SDXLLoRALoaderInvocation"] | components["schemas"]["SDXLModelLoaderInvocation"] | components["schemas"]["SDXLRefinerCompelPromptInvocation"] | components["schemas"]["SDXLRefinerModelLoaderInvocation"] | components["schemas"]["SaveImageInvocation"] | components["schemas"]["ScaleLatentsInvocation"] | components["schemas"]["SchedulerInvocation"] | components["schemas"]["Sd3ModelLoaderInvocation"] | components["schemas"]["Sd3TextEncoderInvocation"] | components["schemas"]["SeamlessModeInvocation"] | components["schemas"]["SegmentAnythingInvocation"] | components["schemas"]["ShowImageInvocation"] | components["schemas"]["SpandrelImageToImageAutoscaleInvocation"] | components["schemas"]["SpandrelImageToImageInvocation"] | components["schemas"]["StringBatchInvocation"] | components["schemas"]["StringCollectionInvocation"] | components["schemas"]["StringGenerator"] | components["schemas"]["StringInvocation"] | components["schemas"]["StringJoinInvocation"] | components["schemas"]["StringJoinThreeInvocation"] | components["schemas"]["StringReplaceInvocation"] | components["schemas"]["StringSplitInvocation"] | components["schemas"]["StringSplitNegInvocation"] | components["schemas"]["SubtractInvocation"] | components["schemas"]["T2IAdapterInvocation"] | components["schemas"]["TileToPropertiesInvocation"] | components["schemas"]["TiledMultiDiffusionDenoiseLatents"] | components["schemas"]["UnsharpMaskInvocation"] | components["schemas"]["VAELoaderInvocation"];
            /**
             * Invocation Source Id
             * @description The ID of the prepared invocation's source node
             */
            invocation_source_id: string;
            /**
             * Error Type
             * @description The error type
             */
            error_type: string;
            /**
             * Error Message
             * @description The error message
             */
            error_message: string;
            /**
             * Error Traceback
             * @description The error traceback
             */
            error_traceback: string;
            /**
             * User Id
             * @description The ID of the user who created the invocation
             * @default null
             */
            user_id: string | null;
            /**
             * Project Id
             * @description The ID of the user who created the invocation
             * @default null
             */
            project_id: string | null;
        };
        InvocationOutputMap: {
            add: components["schemas"]["IntegerOutput"];
            alpha_mask_to_tensor: components["schemas"]["MaskOutput"];
            apply_mask_to_image: components["schemas"]["ImageOutput"];
            apply_tensor_mask_to_image: components["schemas"]["ImageOutput"];
            blank_image: components["schemas"]["ImageOutput"];
            boolean: components["schemas"]["BooleanOutput"];
            boolean_collection: components["schemas"]["BooleanCollectionOutput"];
            bounding_box: components["schemas"]["BoundingBoxOutput"];
            calculate_image_tiles: components["schemas"]["CalculateImageTilesOutput"];
            calculate_image_tiles_even_split: components["schemas"]["CalculateImageTilesOutput"];
            calculate_image_tiles_min_overlap: components["schemas"]["CalculateImageTilesOutput"];
            canny_edge_detection: components["schemas"]["ImageOutput"];
            canvas_paste_back: components["schemas"]["ImageOutput"];
            canvas_v2_mask_and_crop: components["schemas"]["ImageOutput"];
            clip_skip: components["schemas"]["CLIPSkipInvocationOutput"];
            cogview4_denoise: components["schemas"]["LatentsOutput"];
            cogview4_i2l: components["schemas"]["LatentsOutput"];
            cogview4_l2i: components["schemas"]["ImageOutput"];
            cogview4_model_loader: components["schemas"]["CogView4ModelLoaderOutput"];
            cogview4_text_encoder: components["schemas"]["CogView4ConditioningOutput"];
            collect: components["schemas"]["CollectInvocationOutput"];
            color: components["schemas"]["ColorOutput"];
            color_correct: components["schemas"]["ImageOutput"];
            color_map: components["schemas"]["ImageOutput"];
            compel: components["schemas"]["ConditioningOutput"];
            conditioning: components["schemas"]["ConditioningOutput"];
            conditioning_collection: components["schemas"]["ConditioningCollectionOutput"];
            content_shuffle: components["schemas"]["ImageOutput"];
            controlnet: components["schemas"]["ControlOutput"];
            core_metadata: components["schemas"]["MetadataOutput"];
            create_denoise_mask: components["schemas"]["DenoiseMaskOutput"];
            create_gradient_mask: components["schemas"]["GradientMaskOutput"];
            crop_image_to_bounding_box: components["schemas"]["ImageOutput"];
            crop_latents: components["schemas"]["LatentsOutput"];
            cv_inpaint: components["schemas"]["ImageOutput"];
            denoise_latents: components["schemas"]["LatentsOutput"];
            denoise_latents_meta: components["schemas"]["LatentsMetaOutput"];
            depth_anything_depth_estimation: components["schemas"]["ImageOutput"];
            div: components["schemas"]["IntegerOutput"];
            dw_openpose_detection: components["schemas"]["ImageOutput"];
            dynamic_prompt: components["schemas"]["StringCollectionOutput"];
            esrgan: components["schemas"]["ImageOutput"];
            expand_mask_with_fade: components["schemas"]["ImageOutput"];
            face_identifier: components["schemas"]["ImageOutput"];
            face_mask_detection: components["schemas"]["FaceMaskOutput"];
            face_off: components["schemas"]["FaceOffOutput"];
            float: components["schemas"]["FloatOutput"];
            float_batch: components["schemas"]["FloatOutput"];
            float_collection: components["schemas"]["FloatCollectionOutput"];
            float_generator: components["schemas"]["FloatGeneratorOutput"];
            float_math: components["schemas"]["FloatOutput"];
            float_range: components["schemas"]["FloatCollectionOutput"];
            float_to_int: components["schemas"]["IntegerOutput"];
            flux_control_lora_loader: components["schemas"]["FluxControlLoRALoaderOutput"];
            flux_controlnet: components["schemas"]["FluxControlNetOutput"];
            flux_denoise: components["schemas"]["LatentsOutput"];
            flux_denoise_meta: components["schemas"]["LatentsMetaOutput"];
            flux_fill: components["schemas"]["FluxFillOutput"];
            flux_ip_adapter: components["schemas"]["IPAdapterOutput"];
            flux_lora_collection_loader: components["schemas"]["FluxLoRALoaderOutput"];
            flux_lora_loader: components["schemas"]["FluxLoRALoaderOutput"];
            flux_model_loader: components["schemas"]["FluxModelLoaderOutput"];
            flux_redux: components["schemas"]["FluxReduxOutput"];
            flux_text_encoder: components["schemas"]["FluxConditioningOutput"];
            flux_vae_decode: components["schemas"]["ImageOutput"];
            flux_vae_encode: components["schemas"]["LatentsOutput"];
            freeu: components["schemas"]["UNetOutput"];
            get_image_mask_bounding_box: components["schemas"]["BoundingBoxOutput"];
            grounding_dino: components["schemas"]["BoundingBoxCollectionOutput"];
            hed_edge_detection: components["schemas"]["ImageOutput"];
            heuristic_resize: components["schemas"]["ImageOutput"];
            i2l: components["schemas"]["LatentsOutput"];
            ideal_size: components["schemas"]["IdealSizeOutput"];
            image: components["schemas"]["ImageOutput"];
            image_batch: components["schemas"]["ImageOutput"];
            image_collection: components["schemas"]["ImageCollectionOutput"];
            image_generator: components["schemas"]["ImageGeneratorOutput"];
            image_mask_to_tensor: components["schemas"]["MaskOutput"];
            image_panel_layout: components["schemas"]["ImagePanelCoordinateOutput"];
            img_blur: components["schemas"]["ImageOutput"];
            img_chan: components["schemas"]["ImageOutput"];
            img_channel_multiply: components["schemas"]["ImageOutput"];
            img_channel_offset: components["schemas"]["ImageOutput"];
            img_conv: components["schemas"]["ImageOutput"];
            img_crop: components["schemas"]["ImageOutput"];
            img_hue_adjust: components["schemas"]["ImageOutput"];
            img_ilerp: components["schemas"]["ImageOutput"];
            img_lerp: components["schemas"]["ImageOutput"];
            img_mul: components["schemas"]["ImageOutput"];
            img_noise: components["schemas"]["ImageOutput"];
            img_nsfw: components["schemas"]["ImageOutput"];
            img_pad_crop: components["schemas"]["ImageOutput"];
            img_paste: components["schemas"]["ImageOutput"];
            img_resize: components["schemas"]["ImageOutput"];
            img_scale: components["schemas"]["ImageOutput"];
            img_watermark: components["schemas"]["ImageOutput"];
            infill_cv2: components["schemas"]["ImageOutput"];
            infill_lama: components["schemas"]["ImageOutput"];
            infill_patchmatch: components["schemas"]["ImageOutput"];
            infill_rgba: components["schemas"]["ImageOutput"];
            infill_tile: components["schemas"]["ImageOutput"];
            integer: components["schemas"]["IntegerOutput"];
            integer_batch: components["schemas"]["IntegerOutput"];
            integer_collection: components["schemas"]["IntegerCollectionOutput"];
            integer_generator: components["schemas"]["IntegerGeneratorOutput"];
            integer_math: components["schemas"]["IntegerOutput"];
            invert_tensor_mask: components["schemas"]["MaskOutput"];
            invokeai_ealightness: components["schemas"]["ImageOutput"];
            invokeai_img_blend: components["schemas"]["ImageOutput"];
            invokeai_img_composite: components["schemas"]["ImageOutput"];
            invokeai_img_dilate_erode: components["schemas"]["ImageOutput"];
            invokeai_img_enhance: components["schemas"]["ImageOutput"];
            invokeai_img_hue_adjust_plus: components["schemas"]["ImageOutput"];
            invokeai_img_val_thresholds: components["schemas"]["ImageOutput"];
            ip_adapter: components["schemas"]["IPAdapterOutput"];
            iterate: components["schemas"]["IterateInvocationOutput"];
            l2i: components["schemas"]["ImageOutput"];
            latents: components["schemas"]["LatentsOutput"];
            latents_collection: components["schemas"]["LatentsCollectionOutput"];
            lblend: components["schemas"]["LatentsOutput"];
            lineart_anime_edge_detection: components["schemas"]["ImageOutput"];
            lineart_edge_detection: components["schemas"]["ImageOutput"];
            llava_onevision_vllm: components["schemas"]["StringOutput"];
            lora_collection_loader: components["schemas"]["LoRALoaderOutput"];
            lora_loader: components["schemas"]["LoRALoaderOutput"];
            lora_selector: components["schemas"]["LoRASelectorOutput"];
            lresize: components["schemas"]["LatentsOutput"];
            lscale: components["schemas"]["LatentsOutput"];
            main_model_loader: components["schemas"]["ModelLoaderOutput"];
            mask_combine: components["schemas"]["ImageOutput"];
            mask_edge: components["schemas"]["ImageOutput"];
            mask_from_id: components["schemas"]["ImageOutput"];
            mediapipe_face_detection: components["schemas"]["ImageOutput"];
            merge_metadata: components["schemas"]["MetadataOutput"];
            merge_tiles_to_image: components["schemas"]["ImageOutput"];
            metadata: components["schemas"]["MetadataOutput"];
            metadata_field_extractor: components["schemas"]["StringOutput"];
            metadata_from_image: components["schemas"]["MetadataOutput"];
            metadata_item: components["schemas"]["MetadataItemOutput"];
            metadata_item_linked: components["schemas"]["MetadataOutput"];
            metadata_to_bool: components["schemas"]["BooleanOutput"];
            metadata_to_controlnets: components["schemas"]["MDControlListOutput"];
            metadata_to_float: components["schemas"]["FloatOutput"];
            metadata_to_integer: components["schemas"]["IntegerOutput"];
            metadata_to_ip_adapters: components["schemas"]["MDIPAdapterListOutput"];
            metadata_to_lora_collection: components["schemas"]["MetadataToLorasCollectionOutput"];
            metadata_to_loras: components["schemas"]["LoRALoaderOutput"];
            metadata_to_model: components["schemas"]["MetadataToModelOutput"];
            metadata_to_scheduler: components["schemas"]["SchedulerOutput"];
            metadata_to_sdlx_loras: components["schemas"]["SDXLLoRALoaderOutput"];
            metadata_to_sdxl_model: components["schemas"]["MetadataToSDXLModelOutput"];
            metadata_to_string: components["schemas"]["StringOutput"];
            metadata_to_t2i_adapters: components["schemas"]["MDT2IAdapterListOutput"];
            metadata_to_vae: components["schemas"]["VAEOutput"];
            mlsd_detection: components["schemas"]["ImageOutput"];
            model_identifier: components["schemas"]["ModelIdentifierOutput"];
            mul: components["schemas"]["IntegerOutput"];
            noise: components["schemas"]["NoiseOutput"];
            normal_map: components["schemas"]["ImageOutput"];
            pair_tile_image: components["schemas"]["PairTileImageOutput"];
            paste_image_into_bounding_box: components["schemas"]["ImageOutput"];
            pidi_edge_detection: components["schemas"]["ImageOutput"];
            prompt_from_file: components["schemas"]["StringCollectionOutput"];
            rand_float: components["schemas"]["FloatOutput"];
            rand_int: components["schemas"]["IntegerOutput"];
            random_range: components["schemas"]["IntegerCollectionOutput"];
            range: components["schemas"]["IntegerCollectionOutput"];
            range_of_size: components["schemas"]["IntegerCollectionOutput"];
            rectangle_mask: components["schemas"]["MaskOutput"];
            round_float: components["schemas"]["FloatOutput"];
            save_image: components["schemas"]["ImageOutput"];
            scheduler: components["schemas"]["SchedulerOutput"];
            sd3_denoise: components["schemas"]["LatentsOutput"];
            sd3_i2l: components["schemas"]["LatentsOutput"];
            sd3_l2i: components["schemas"]["ImageOutput"];
            sd3_model_loader: components["schemas"]["Sd3ModelLoaderOutput"];
            sd3_text_encoder: components["schemas"]["SD3ConditioningOutput"];
            sdxl_compel_prompt: components["schemas"]["ConditioningOutput"];
            sdxl_lora_collection_loader: components["schemas"]["SDXLLoRALoaderOutput"];
            sdxl_lora_loader: components["schemas"]["SDXLLoRALoaderOutput"];
            sdxl_model_loader: components["schemas"]["SDXLModelLoaderOutput"];
            sdxl_refiner_compel_prompt: components["schemas"]["ConditioningOutput"];
            sdxl_refiner_model_loader: components["schemas"]["SDXLRefinerModelLoaderOutput"];
            seamless: components["schemas"]["SeamlessModeOutput"];
            segment_anything: components["schemas"]["MaskOutput"];
            show_image: components["schemas"]["ImageOutput"];
            spandrel_image_to_image: components["schemas"]["ImageOutput"];
            spandrel_image_to_image_autoscale: components["schemas"]["ImageOutput"];
            string: components["schemas"]["StringOutput"];
            string_batch: components["schemas"]["StringOutput"];
            string_collection: components["schemas"]["StringCollectionOutput"];
            string_generator: components["schemas"]["StringGeneratorOutput"];
            string_join: components["schemas"]["StringOutput"];
            string_join_three: components["schemas"]["StringOutput"];
            string_replace: components["schemas"]["StringOutput"];
            string_split: components["schemas"]["String2Output"];
            string_split_neg: components["schemas"]["StringPosNegOutput"];
            sub: components["schemas"]["IntegerOutput"];
            t2i_adapter: components["schemas"]["T2IAdapterOutput"];
            tensor_mask_to_image: components["schemas"]["ImageOutput"];
            tile_to_properties: components["schemas"]["TileToPropertiesOutput"];
            tiled_multi_diffusion_denoise_latents: components["schemas"]["LatentsOutput"];
            tomask: components["schemas"]["ImageOutput"];
            unsharp_mask: components["schemas"]["ImageOutput"];
            vae_loader: components["schemas"]["VAEOutput"];
        };
        /**
         * InvocationProgressEvent
         * @description Event model for invocation_progress
         */
        InvocationProgressEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Item Id
             * @description The ID of the queue item
             */
            item_id: number;
            /**
             * Batch Id
             * @description The ID of the queue batch
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of the queue item
             * @default null
             */
            origin: string | null;
            /**
             * Destination
             * @description The destination of the queue item
             * @default null
             */
            destination: string | null;
            /**
             * Session Id
             * @description The ID of the session (aka graph execution state)
             */
            session_id: string;
            /**
             * Invocation
             * @description The ID of the invocation
             */
            invocation: components["schemas"]["AddInvocation"] | components["schemas"]["AlphaMaskToTensorInvocation"] | components["schemas"]["ApplyMaskTensorToImageInvocation"] | components["schemas"]["ApplyMaskToImageInvocation"] | components["schemas"]["BlankImageInvocation"] | components["schemas"]["BlendLatentsInvocation"] | components["schemas"]["BooleanCollectionInvocation"] | components["schemas"]["BooleanInvocation"] | components["schemas"]["BoundingBoxInvocation"] | components["schemas"]["CLIPSkipInvocation"] | components["schemas"]["CV2InfillInvocation"] | components["schemas"]["CalculateImageTilesEvenSplitInvocation"] | components["schemas"]["CalculateImageTilesInvocation"] | components["schemas"]["CalculateImageTilesMinimumOverlapInvocation"] | components["schemas"]["CannyEdgeDetectionInvocation"] | components["schemas"]["CanvasPasteBackInvocation"] | components["schemas"]["CanvasV2MaskAndCropInvocation"] | components["schemas"]["CenterPadCropInvocation"] | components["schemas"]["CogView4DenoiseInvocation"] | components["schemas"]["CogView4ImageToLatentsInvocation"] | components["schemas"]["CogView4LatentsToImageInvocation"] | components["schemas"]["CogView4ModelLoaderInvocation"] | components["schemas"]["CogView4TextEncoderInvocation"] | components["schemas"]["CollectInvocation"] | components["schemas"]["ColorCorrectInvocation"] | components["schemas"]["ColorInvocation"] | components["schemas"]["ColorMapInvocation"] | components["schemas"]["CompelInvocation"] | components["schemas"]["ConditioningCollectionInvocation"] | components["schemas"]["ConditioningInvocation"] | components["schemas"]["ContentShuffleInvocation"] | components["schemas"]["ControlNetInvocation"] | components["schemas"]["CoreMetadataInvocation"] | components["schemas"]["CreateDenoiseMaskInvocation"] | components["schemas"]["CreateGradientMaskInvocation"] | components["schemas"]["CropImageToBoundingBoxInvocation"] | components["schemas"]["CropLatentsCoreInvocation"] | components["schemas"]["CvInpaintInvocation"] | components["schemas"]["DWOpenposeDetectionInvocation"] | components["schemas"]["DenoiseLatentsInvocation"] | components["schemas"]["DenoiseLatentsMetaInvocation"] | components["schemas"]["DepthAnythingDepthEstimationInvocation"] | components["schemas"]["DivideInvocation"] | components["schemas"]["DynamicPromptInvocation"] | components["schemas"]["ESRGANInvocation"] | components["schemas"]["ExpandMaskWithFadeInvocation"] | components["schemas"]["FLUXLoRACollectionLoader"] | components["schemas"]["FaceIdentifierInvocation"] | components["schemas"]["FaceMaskInvocation"] | components["schemas"]["FaceOffInvocation"] | components["schemas"]["FloatBatchInvocation"] | components["schemas"]["FloatCollectionInvocation"] | components["schemas"]["FloatGenerator"] | components["schemas"]["FloatInvocation"] | components["schemas"]["FloatLinearRangeInvocation"] | components["schemas"]["FloatMathInvocation"] | components["schemas"]["FloatToIntegerInvocation"] | components["schemas"]["FluxControlLoRALoaderInvocation"] | components["schemas"]["FluxControlNetInvocation"] | components["schemas"]["FluxDenoiseInvocation"] | components["schemas"]["FluxDenoiseLatentsMetaInvocation"] | components["schemas"]["FluxFillInvocation"] | components["schemas"]["FluxIPAdapterInvocation"] | components["schemas"]["FluxLoRALoaderInvocation"] | components["schemas"]["FluxModelLoaderInvocation"] | components["schemas"]["FluxReduxInvocation"] | components["schemas"]["FluxTextEncoderInvocation"] | components["schemas"]["FluxVaeDecodeInvocation"] | components["schemas"]["FluxVaeEncodeInvocation"] | components["schemas"]["FreeUInvocation"] | components["schemas"]["GetMaskBoundingBoxInvocation"] | components["schemas"]["GroundingDinoInvocation"] | components["schemas"]["HEDEdgeDetectionInvocation"] | components["schemas"]["HeuristicResizeInvocation"] | components["schemas"]["IPAdapterInvocation"] | components["schemas"]["IdealSizeInvocation"] | components["schemas"]["ImageBatchInvocation"] | components["schemas"]["ImageBlurInvocation"] | components["schemas"]["ImageChannelInvocation"] | components["schemas"]["ImageChannelMultiplyInvocation"] | components["schemas"]["ImageChannelOffsetInvocation"] | components["schemas"]["ImageCollectionInvocation"] | components["schemas"]["ImageConvertInvocation"] | components["schemas"]["ImageCropInvocation"] | components["schemas"]["ImageGenerator"] | components["schemas"]["ImageHueAdjustmentInvocation"] | components["schemas"]["ImageInverseLerpInvocation"] | components["schemas"]["ImageInvocation"] | components["schemas"]["ImageLerpInvocation"] | components["schemas"]["ImageMaskToTensorInvocation"] | components["schemas"]["ImageMultiplyInvocation"] | components["schemas"]["ImageNSFWBlurInvocation"] | components["schemas"]["ImageNoiseInvocation"] | components["schemas"]["ImagePanelLayoutInvocation"] | components["schemas"]["ImagePasteInvocation"] | components["schemas"]["ImageResizeInvocation"] | components["schemas"]["ImageScaleInvocation"] | components["schemas"]["ImageToLatentsInvocation"] | components["schemas"]["ImageWatermarkInvocation"] | components["schemas"]["InfillColorInvocation"] | components["schemas"]["InfillPatchMatchInvocation"] | components["schemas"]["InfillTileInvocation"] | components["schemas"]["IntegerBatchInvocation"] | components["schemas"]["IntegerCollectionInvocation"] | components["schemas"]["IntegerGenerator"] | components["schemas"]["IntegerInvocation"] | components["schemas"]["IntegerMathInvocation"] | components["schemas"]["InvertTensorMaskInvocation"] | components["schemas"]["InvokeAdjustImageHuePlusInvocation"] | components["schemas"]["InvokeEquivalentAchromaticLightnessInvocation"] | components["schemas"]["InvokeImageBlendInvocation"] | components["schemas"]["InvokeImageCompositorInvocation"] | components["schemas"]["InvokeImageDilateOrErodeInvocation"] | components["schemas"]["InvokeImageEnhanceInvocation"] | components["schemas"]["InvokeImageValueThresholdsInvocation"] | components["schemas"]["IterateInvocation"] | components["schemas"]["LaMaInfillInvocation"] | components["schemas"]["LatentsCollectionInvocation"] | components["schemas"]["LatentsInvocation"] | components["schemas"]["LatentsToImageInvocation"] | components["schemas"]["LineartAnimeEdgeDetectionInvocation"] | components["schemas"]["LineartEdgeDetectionInvocation"] | components["schemas"]["LlavaOnevisionVllmInvocation"] | components["schemas"]["LoRACollectionLoader"] | components["schemas"]["LoRALoaderInvocation"] | components["schemas"]["LoRASelectorInvocation"] | components["schemas"]["MLSDDetectionInvocation"] | components["schemas"]["MainModelLoaderInvocation"] | components["schemas"]["MaskCombineInvocation"] | components["schemas"]["MaskEdgeInvocation"] | components["schemas"]["MaskFromAlphaInvocation"] | components["schemas"]["MaskFromIDInvocation"] | components["schemas"]["MaskTensorToImageInvocation"] | components["schemas"]["MediaPipeFaceDetectionInvocation"] | components["schemas"]["MergeMetadataInvocation"] | components["schemas"]["MergeTilesToImageInvocation"] | components["schemas"]["MetadataFieldExtractorInvocation"] | components["schemas"]["MetadataFromImageInvocation"] | components["schemas"]["MetadataInvocation"] | components["schemas"]["MetadataItemInvocation"] | components["schemas"]["MetadataItemLinkedInvocation"] | components["schemas"]["MetadataToBoolInvocation"] | components["schemas"]["MetadataToControlnetsInvocation"] | components["schemas"]["MetadataToFloatInvocation"] | components["schemas"]["MetadataToIPAdaptersInvocation"] | components["schemas"]["MetadataToIntegerInvocation"] | components["schemas"]["MetadataToLorasCollectionInvocation"] | components["schemas"]["MetadataToLorasInvocation"] | components["schemas"]["MetadataToModelInvocation"] | components["schemas"]["MetadataToSDXLLorasInvocation"] | components["schemas"]["MetadataToSDXLModelInvocation"] | components["schemas"]["MetadataToSchedulerInvocation"] | components["schemas"]["MetadataToStringInvocation"] | components["schemas"]["MetadataToT2IAdaptersInvocation"] | components["schemas"]["MetadataToVAEInvocation"] | components["schemas"]["ModelIdentifierInvocation"] | components["schemas"]["MultiplyInvocation"] | components["schemas"]["NoiseInvocation"] | components["schemas"]["NormalMapInvocation"] | components["schemas"]["PairTileImageInvocation"] | components["schemas"]["PasteImageIntoBoundingBoxInvocation"] | components["schemas"]["PiDiNetEdgeDetectionInvocation"] | components["schemas"]["PromptsFromFileInvocation"] | components["schemas"]["RandomFloatInvocation"] | components["schemas"]["RandomIntInvocation"] | components["schemas"]["RandomRangeInvocation"] | components["schemas"]["RangeInvocation"] | components["schemas"]["RangeOfSizeInvocation"] | components["schemas"]["RectangleMaskInvocation"] | components["schemas"]["ResizeLatentsInvocation"] | components["schemas"]["RoundInvocation"] | components["schemas"]["SD3DenoiseInvocation"] | components["schemas"]["SD3ImageToLatentsInvocation"] | components["schemas"]["SD3LatentsToImageInvocation"] | components["schemas"]["SDXLCompelPromptInvocation"] | components["schemas"]["SDXLLoRACollectionLoader"] | components["schemas"]["SDXLLoRALoaderInvocation"] | components["schemas"]["SDXLModelLoaderInvocation"] | components["schemas"]["SDXLRefinerCompelPromptInvocation"] | components["schemas"]["SDXLRefinerModelLoaderInvocation"] | components["schemas"]["SaveImageInvocation"] | components["schemas"]["ScaleLatentsInvocation"] | components["schemas"]["SchedulerInvocation"] | components["schemas"]["Sd3ModelLoaderInvocation"] | components["schemas"]["Sd3TextEncoderInvocation"] | components["schemas"]["SeamlessModeInvocation"] | components["schemas"]["SegmentAnythingInvocation"] | components["schemas"]["ShowImageInvocation"] | components["schemas"]["SpandrelImageToImageAutoscaleInvocation"] | components["schemas"]["SpandrelImageToImageInvocation"] | components["schemas"]["StringBatchInvocation"] | components["schemas"]["StringCollectionInvocation"] | components["schemas"]["StringGenerator"] | components["schemas"]["StringInvocation"] | components["schemas"]["StringJoinInvocation"] | components["schemas"]["StringJoinThreeInvocation"] | components["schemas"]["StringReplaceInvocation"] | components["schemas"]["StringSplitInvocation"] | components["schemas"]["StringSplitNegInvocation"] | components["schemas"]["SubtractInvocation"] | components["schemas"]["T2IAdapterInvocation"] | components["schemas"]["TileToPropertiesInvocation"] | components["schemas"]["TiledMultiDiffusionDenoiseLatents"] | components["schemas"]["UnsharpMaskInvocation"] | components["schemas"]["VAELoaderInvocation"];
            /**
             * Invocation Source Id
             * @description The ID of the prepared invocation's source node
             */
            invocation_source_id: string;
            /**
             * Message
             * @description A message to display
             */
            message: string;
            /**
             * Percentage
             * @description The percentage of the progress (omit to indicate indeterminate progress)
             * @default null
             */
            percentage: number | null;
            /**
             * @description An image representing the current state of the progress
             * @default null
             */
            image: components["schemas"]["ProgressImage"] | null;
        };
        /**
         * InvocationStartedEvent
         * @description Event model for invocation_started
         */
        InvocationStartedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Item Id
             * @description The ID of the queue item
             */
            item_id: number;
            /**
             * Batch Id
             * @description The ID of the queue batch
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of the queue item
             * @default null
             */
            origin: string | null;
            /**
             * Destination
             * @description The destination of the queue item
             * @default null
             */
            destination: string | null;
            /**
             * Session Id
             * @description The ID of the session (aka graph execution state)
             */
            session_id: string;
            /**
             * Invocation
             * @description The ID of the invocation
             */
            invocation: components["schemas"]["AddInvocation"] | components["schemas"]["AlphaMaskToTensorInvocation"] | components["schemas"]["ApplyMaskTensorToImageInvocation"] | components["schemas"]["ApplyMaskToImageInvocation"] | components["schemas"]["BlankImageInvocation"] | components["schemas"]["BlendLatentsInvocation"] | components["schemas"]["BooleanCollectionInvocation"] | components["schemas"]["BooleanInvocation"] | components["schemas"]["BoundingBoxInvocation"] | components["schemas"]["CLIPSkipInvocation"] | components["schemas"]["CV2InfillInvocation"] | components["schemas"]["CalculateImageTilesEvenSplitInvocation"] | components["schemas"]["CalculateImageTilesInvocation"] | components["schemas"]["CalculateImageTilesMinimumOverlapInvocation"] | components["schemas"]["CannyEdgeDetectionInvocation"] | components["schemas"]["CanvasPasteBackInvocation"] | components["schemas"]["CanvasV2MaskAndCropInvocation"] | components["schemas"]["CenterPadCropInvocation"] | components["schemas"]["CogView4DenoiseInvocation"] | components["schemas"]["CogView4ImageToLatentsInvocation"] | components["schemas"]["CogView4LatentsToImageInvocation"] | components["schemas"]["CogView4ModelLoaderInvocation"] | components["schemas"]["CogView4TextEncoderInvocation"] | components["schemas"]["CollectInvocation"] | components["schemas"]["ColorCorrectInvocation"] | components["schemas"]["ColorInvocation"] | components["schemas"]["ColorMapInvocation"] | components["schemas"]["CompelInvocation"] | components["schemas"]["ConditioningCollectionInvocation"] | components["schemas"]["ConditioningInvocation"] | components["schemas"]["ContentShuffleInvocation"] | components["schemas"]["ControlNetInvocation"] | components["schemas"]["CoreMetadataInvocation"] | components["schemas"]["CreateDenoiseMaskInvocation"] | components["schemas"]["CreateGradientMaskInvocation"] | components["schemas"]["CropImageToBoundingBoxInvocation"] | components["schemas"]["CropLatentsCoreInvocation"] | components["schemas"]["CvInpaintInvocation"] | components["schemas"]["DWOpenposeDetectionInvocation"] | components["schemas"]["DenoiseLatentsInvocation"] | components["schemas"]["DenoiseLatentsMetaInvocation"] | components["schemas"]["DepthAnythingDepthEstimationInvocation"] | components["schemas"]["DivideInvocation"] | components["schemas"]["DynamicPromptInvocation"] | components["schemas"]["ESRGANInvocation"] | components["schemas"]["ExpandMaskWithFadeInvocation"] | components["schemas"]["FLUXLoRACollectionLoader"] | components["schemas"]["FaceIdentifierInvocation"] | components["schemas"]["FaceMaskInvocation"] | components["schemas"]["FaceOffInvocation"] | components["schemas"]["FloatBatchInvocation"] | components["schemas"]["FloatCollectionInvocation"] | components["schemas"]["FloatGenerator"] | components["schemas"]["FloatInvocation"] | components["schemas"]["FloatLinearRangeInvocation"] | components["schemas"]["FloatMathInvocation"] | components["schemas"]["FloatToIntegerInvocation"] | components["schemas"]["FluxControlLoRALoaderInvocation"] | components["schemas"]["FluxControlNetInvocation"] | components["schemas"]["FluxDenoiseInvocation"] | components["schemas"]["FluxDenoiseLatentsMetaInvocation"] | components["schemas"]["FluxFillInvocation"] | components["schemas"]["FluxIPAdapterInvocation"] | components["schemas"]["FluxLoRALoaderInvocation"] | components["schemas"]["FluxModelLoaderInvocation"] | components["schemas"]["FluxReduxInvocation"] | components["schemas"]["FluxTextEncoderInvocation"] | components["schemas"]["FluxVaeDecodeInvocation"] | components["schemas"]["FluxVaeEncodeInvocation"] | components["schemas"]["FreeUInvocation"] | components["schemas"]["GetMaskBoundingBoxInvocation"] | components["schemas"]["GroundingDinoInvocation"] | components["schemas"]["HEDEdgeDetectionInvocation"] | components["schemas"]["HeuristicResizeInvocation"] | components["schemas"]["IPAdapterInvocation"] | components["schemas"]["IdealSizeInvocation"] | components["schemas"]["ImageBatchInvocation"] | components["schemas"]["ImageBlurInvocation"] | components["schemas"]["ImageChannelInvocation"] | components["schemas"]["ImageChannelMultiplyInvocation"] | components["schemas"]["ImageChannelOffsetInvocation"] | components["schemas"]["ImageCollectionInvocation"] | components["schemas"]["ImageConvertInvocation"] | components["schemas"]["ImageCropInvocation"] | components["schemas"]["ImageGenerator"] | components["schemas"]["ImageHueAdjustmentInvocation"] | components["schemas"]["ImageInverseLerpInvocation"] | components["schemas"]["ImageInvocation"] | components["schemas"]["ImageLerpInvocation"] | components["schemas"]["ImageMaskToTensorInvocation"] | components["schemas"]["ImageMultiplyInvocation"] | components["schemas"]["ImageNSFWBlurInvocation"] | components["schemas"]["ImageNoiseInvocation"] | components["schemas"]["ImagePanelLayoutInvocation"] | components["schemas"]["ImagePasteInvocation"] | components["schemas"]["ImageResizeInvocation"] | components["schemas"]["ImageScaleInvocation"] | components["schemas"]["ImageToLatentsInvocation"] | components["schemas"]["ImageWatermarkInvocation"] | components["schemas"]["InfillColorInvocation"] | components["schemas"]["InfillPatchMatchInvocation"] | components["schemas"]["InfillTileInvocation"] | components["schemas"]["IntegerBatchInvocation"] | components["schemas"]["IntegerCollectionInvocation"] | components["schemas"]["IntegerGenerator"] | components["schemas"]["IntegerInvocation"] | components["schemas"]["IntegerMathInvocation"] | components["schemas"]["InvertTensorMaskInvocation"] | components["schemas"]["InvokeAdjustImageHuePlusInvocation"] | components["schemas"]["InvokeEquivalentAchromaticLightnessInvocation"] | components["schemas"]["InvokeImageBlendInvocation"] | components["schemas"]["InvokeImageCompositorInvocation"] | components["schemas"]["InvokeImageDilateOrErodeInvocation"] | components["schemas"]["InvokeImageEnhanceInvocation"] | components["schemas"]["InvokeImageValueThresholdsInvocation"] | components["schemas"]["IterateInvocation"] | components["schemas"]["LaMaInfillInvocation"] | components["schemas"]["LatentsCollectionInvocation"] | components["schemas"]["LatentsInvocation"] | components["schemas"]["LatentsToImageInvocation"] | components["schemas"]["LineartAnimeEdgeDetectionInvocation"] | components["schemas"]["LineartEdgeDetectionInvocation"] | components["schemas"]["LlavaOnevisionVllmInvocation"] | components["schemas"]["LoRACollectionLoader"] | components["schemas"]["LoRALoaderInvocation"] | components["schemas"]["LoRASelectorInvocation"] | components["schemas"]["MLSDDetectionInvocation"] | components["schemas"]["MainModelLoaderInvocation"] | components["schemas"]["MaskCombineInvocation"] | components["schemas"]["MaskEdgeInvocation"] | components["schemas"]["MaskFromAlphaInvocation"] | components["schemas"]["MaskFromIDInvocation"] | components["schemas"]["MaskTensorToImageInvocation"] | components["schemas"]["MediaPipeFaceDetectionInvocation"] | components["schemas"]["MergeMetadataInvocation"] | components["schemas"]["MergeTilesToImageInvocation"] | components["schemas"]["MetadataFieldExtractorInvocation"] | components["schemas"]["MetadataFromImageInvocation"] | components["schemas"]["MetadataInvocation"] | components["schemas"]["MetadataItemInvocation"] | components["schemas"]["MetadataItemLinkedInvocation"] | components["schemas"]["MetadataToBoolInvocation"] | components["schemas"]["MetadataToControlnetsInvocation"] | components["schemas"]["MetadataToFloatInvocation"] | components["schemas"]["MetadataToIPAdaptersInvocation"] | components["schemas"]["MetadataToIntegerInvocation"] | components["schemas"]["MetadataToLorasCollectionInvocation"] | components["schemas"]["MetadataToLorasInvocation"] | components["schemas"]["MetadataToModelInvocation"] | components["schemas"]["MetadataToSDXLLorasInvocation"] | components["schemas"]["MetadataToSDXLModelInvocation"] | components["schemas"]["MetadataToSchedulerInvocation"] | components["schemas"]["MetadataToStringInvocation"] | components["schemas"]["MetadataToT2IAdaptersInvocation"] | components["schemas"]["MetadataToVAEInvocation"] | components["schemas"]["ModelIdentifierInvocation"] | components["schemas"]["MultiplyInvocation"] | components["schemas"]["NoiseInvocation"] | components["schemas"]["NormalMapInvocation"] | components["schemas"]["PairTileImageInvocation"] | components["schemas"]["PasteImageIntoBoundingBoxInvocation"] | components["schemas"]["PiDiNetEdgeDetectionInvocation"] | components["schemas"]["PromptsFromFileInvocation"] | components["schemas"]["RandomFloatInvocation"] | components["schemas"]["RandomIntInvocation"] | components["schemas"]["RandomRangeInvocation"] | components["schemas"]["RangeInvocation"] | components["schemas"]["RangeOfSizeInvocation"] | components["schemas"]["RectangleMaskInvocation"] | components["schemas"]["ResizeLatentsInvocation"] | components["schemas"]["RoundInvocation"] | components["schemas"]["SD3DenoiseInvocation"] | components["schemas"]["SD3ImageToLatentsInvocation"] | components["schemas"]["SD3LatentsToImageInvocation"] | components["schemas"]["SDXLCompelPromptInvocation"] | components["schemas"]["SDXLLoRACollectionLoader"] | components["schemas"]["SDXLLoRALoaderInvocation"] | components["schemas"]["SDXLModelLoaderInvocation"] | components["schemas"]["SDXLRefinerCompelPromptInvocation"] | components["schemas"]["SDXLRefinerModelLoaderInvocation"] | components["schemas"]["SaveImageInvocation"] | components["schemas"]["ScaleLatentsInvocation"] | components["schemas"]["SchedulerInvocation"] | components["schemas"]["Sd3ModelLoaderInvocation"] | components["schemas"]["Sd3TextEncoderInvocation"] | components["schemas"]["SeamlessModeInvocation"] | components["schemas"]["SegmentAnythingInvocation"] | components["schemas"]["ShowImageInvocation"] | components["schemas"]["SpandrelImageToImageAutoscaleInvocation"] | components["schemas"]["SpandrelImageToImageInvocation"] | components["schemas"]["StringBatchInvocation"] | components["schemas"]["StringCollectionInvocation"] | components["schemas"]["StringGenerator"] | components["schemas"]["StringInvocation"] | components["schemas"]["StringJoinInvocation"] | components["schemas"]["StringJoinThreeInvocation"] | components["schemas"]["StringReplaceInvocation"] | components["schemas"]["StringSplitInvocation"] | components["schemas"]["StringSplitNegInvocation"] | components["schemas"]["SubtractInvocation"] | components["schemas"]["T2IAdapterInvocation"] | components["schemas"]["TileToPropertiesInvocation"] | components["schemas"]["TiledMultiDiffusionDenoiseLatents"] | components["schemas"]["UnsharpMaskInvocation"] | components["schemas"]["VAELoaderInvocation"];
            /**
             * Invocation Source Id
             * @description The ID of the prepared invocation's source node
             */
            invocation_source_id: string;
        };
        /**
         * InvokeAIAppConfig
         * @description Invoke's global app configuration.
         *
         *     Typically, you won't need to interact with this class directly. Instead, use the `get_config` function from `invokeai.app.services.config` to get a singleton config object.
         *
         *     Attributes:
         *         host: IP address to bind to. Use `0.0.0.0` to serve to your local network.
         *         port: Port to bind to.
         *         allow_origins: Allowed CORS origins.
         *         allow_credentials: Allow CORS credentials.
         *         allow_methods: Methods allowed for CORS.
         *         allow_headers: Headers allowed for CORS.
         *         ssl_certfile: SSL certificate file for HTTPS. See https://www.uvicorn.org/settings/#https.
         *         ssl_keyfile: SSL key file for HTTPS. See https://www.uvicorn.org/settings/#https.
         *         log_tokenization: Enable logging of parsed prompt tokens.
         *         patchmatch: Enable patchmatch inpaint code.
         *         models_dir: Path to the models directory.
         *         convert_cache_dir: Path to the converted models cache directory (DEPRECATED, but do not delete because it is needed for migration from previous versions).
         *         download_cache_dir: Path to the directory that contains dynamically downloaded models.
         *         legacy_conf_dir: Path to directory of legacy checkpoint config files.
         *         db_dir: Path to InvokeAI databases directory.
         *         outputs_dir: Path to directory for outputs.
         *         custom_nodes_dir: Path to directory for custom nodes.
         *         style_presets_dir: Path to directory for style presets.
         *         workflow_thumbnails_dir: Path to directory for workflow thumbnails.
         *         log_handlers: Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".
         *         log_format: Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.<br>Valid values: `plain`, `color`, `syslog`, `legacy`
         *         log_level: Emit logging messages at this level or higher.<br>Valid values: `debug`, `info`, `warning`, `error`, `critical`
         *         log_sql: Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.
         *         log_level_network: Log level for network-related messages. 'info' and 'debug' are very verbose.<br>Valid values: `debug`, `info`, `warning`, `error`, `critical`
         *         use_memory_db: Use in-memory database. Useful for development.
         *         dev_reload: Automatically reload when Python sources are changed. Does not reload node definitions.
         *         profile_graphs: Enable graph profiling using `cProfile`.
         *         profile_prefix: An optional prefix for profile output files.
         *         profiles_dir: Path to profiles output directory.
         *         max_cache_ram_gb: The maximum amount of CPU RAM to use for model caching in GB. If unset, the limit will be configured based on the available RAM. In most cases, it is recommended to leave this unset.
         *         max_cache_vram_gb: The amount of VRAM to use for model caching in GB. If unset, the limit will be configured based on the available VRAM and the device_working_mem_gb. In most cases, it is recommended to leave this unset.
         *         log_memory_usage: If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.
         *         device_working_mem_gb: The amount of working memory to keep available on the compute device (in GB). Has no effect if running on CPU. If you are experiencing OOM errors, try increasing this value.
         *         enable_partial_loading: Enable partial loading of models. This enables models to run with reduced VRAM requirements (at the cost of slower speed) by streaming the model from RAM to VRAM as its used. In some edge cases, partial loading can cause models to run more slowly if they were previously being fully loaded into VRAM.
         *         keep_ram_copy_of_weights: Whether to keep a full RAM copy of a model's weights when the model is loaded in VRAM. Keeping a RAM copy increases average RAM usage, but speeds up model switching and LoRA patching (assuming there is sufficient RAM). Set this to False if RAM pressure is consistently high.
         *         ram: DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_ram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.
         *         vram: DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_vram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.
         *         lazy_offload: DEPRECATED: This setting is no longer used. Lazy-offloading is enabled by default. This config setting will be removed once the new model cache behavior is stable.
         *         pytorch_cuda_alloc_conf: Configure the Torch CUDA memory allocator. This will impact peak reserved VRAM usage and performance. Setting to "backend:cudaMallocAsync" works well on many systems. The optimal configuration is highly dependent on the system configuration (device type, VRAM, CUDA driver version, etc.), so must be tuned experimentally.
         *         device: Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.<br>Valid values: `auto`, `cpu`, `cuda`, `cuda:1`, `mps`
         *         precision: Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.<br>Valid values: `auto`, `float16`, `bfloat16`, `float32`
         *         sequential_guidance: Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.
         *         attention_type: Attention type.<br>Valid values: `auto`, `normal`, `xformers`, `sliced`, `torch-sdp`
         *         attention_slice_size: Slice size, valid when attention_type=="sliced".<br>Valid values: `auto`, `balanced`, `max`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`
         *         force_tiled_decode: Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).
         *         pil_compress_level: The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.
         *         max_queue_size: Maximum number of items in the session queue.
         *         clear_queue_on_startup: Empties session queue on startup.
         *         allow_nodes: List of nodes to allow. Omit to allow all.
         *         deny_nodes: List of nodes to deny. Omit to deny none.
         *         node_cache_size: How many cached nodes to keep in memory.
         *         hashing_algorithm: Model hashing algorthim for model installs. 'blake3_multi' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.<br>Valid values: `blake3_multi`, `blake3_single`, `random`, `md5`, `sha1`, `sha224`, `sha256`, `sha384`, `sha512`, `blake2b`, `blake2s`, `sha3_224`, `sha3_256`, `sha3_384`, `sha3_512`, `shake_128`, `shake_256`
         *         remote_api_tokens: List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.
         *         scan_models_on_startup: Scan the models directory on startup, registering orphaned models. This is typically only used in conjunction with `use_memory_db` for testing purposes.
         */
        InvokeAIAppConfig: {
            /**
             * Schema Version
             * @description Schema version of the config file. This is not a user-configurable setting.
             * @default 4.0.2
             */
            schema_version?: string;
            /**
             * Legacy Models Yaml Path
             * @description Path to the legacy models.yaml file. This is not a user-configurable setting.
             */
            legacy_models_yaml_path?: string | null;
            /**
             * Host
             * @description IP address to bind to. Use `0.0.0.0` to serve to your local network.
             * @default 127.0.0.1
             */
            host?: string;
            /**
             * Port
             * @description Port to bind to.
             * @default 9090
             */
            port?: number;
            /**
             * Allow Origins
             * @description Allowed CORS origins.
             * @default []
             */
            allow_origins?: string[];
            /**
             * Allow Credentials
             * @description Allow CORS credentials.
             * @default true
             */
            allow_credentials?: boolean;
            /**
             * Allow Methods
             * @description Methods allowed for CORS.
             * @default [
             *       "*"
             *     ]
             */
            allow_methods?: string[];
            /**
             * Allow Headers
             * @description Headers allowed for CORS.
             * @default [
             *       "*"
             *     ]
             */
            allow_headers?: string[];
            /**
             * Ssl Certfile
             * @description SSL certificate file for HTTPS. See https://www.uvicorn.org/settings/#https.
             */
            ssl_certfile?: string | null;
            /**
             * Ssl Keyfile
             * @description SSL key file for HTTPS. See https://www.uvicorn.org/settings/#https.
             */
            ssl_keyfile?: string | null;
            /**
             * Log Tokenization
             * @description Enable logging of parsed prompt tokens.
             * @default false
             */
            log_tokenization?: boolean;
            /**
             * Patchmatch
             * @description Enable patchmatch inpaint code.
             * @default true
             */
            patchmatch?: boolean;
            /**
             * Models Dir
             * Format: path
             * @description Path to the models directory.
             * @default models
             */
            models_dir?: string;
            /**
             * Convert Cache Dir
             * Format: path
             * @description Path to the converted models cache directory (DEPRECATED, but do not delete because it is needed for migration from previous versions).
             * @default models/.convert_cache
             */
            convert_cache_dir?: string;
            /**
             * Download Cache Dir
             * Format: path
             * @description Path to the directory that contains dynamically downloaded models.
             * @default models/.download_cache
             */
            download_cache_dir?: string;
            /**
             * Legacy Conf Dir
             * Format: path
             * @description Path to directory of legacy checkpoint config files.
             * @default configs
             */
            legacy_conf_dir?: string;
            /**
             * Db Dir
             * Format: path
             * @description Path to InvokeAI databases directory.
             * @default databases
             */
            db_dir?: string;
            /**
             * Outputs Dir
             * Format: path
             * @description Path to directory for outputs.
             * @default outputs
             */
            outputs_dir?: string;
            /**
             * Custom Nodes Dir
             * Format: path
             * @description Path to directory for custom nodes.
             * @default nodes
             */
            custom_nodes_dir?: string;
            /**
             * Style Presets Dir
             * Format: path
             * @description Path to directory for style presets.
             * @default style_presets
             */
            style_presets_dir?: string;
            /**
             * Workflow Thumbnails Dir
             * Format: path
             * @description Path to directory for workflow thumbnails.
             * @default workflow_thumbnails
             */
            workflow_thumbnails_dir?: string;
            /**
             * Log Handlers
             * @description Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".
             * @default [
             *       "console"
             *     ]
             */
            log_handlers?: string[];
            /**
             * Log Format
             * @description Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.
             * @default color
             * @enum {string}
             */
            log_format?: "plain" | "color" | "syslog" | "legacy";
            /**
             * Log Level
             * @description Emit logging messages at this level or higher.
             * @default info
             * @enum {string}
             */
            log_level?: "debug" | "info" | "warning" | "error" | "critical";
            /**
             * Log Sql
             * @description Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.
             * @default false
             */
            log_sql?: boolean;
            /**
             * Log Level Network
             * @description Log level for network-related messages. 'info' and 'debug' are very verbose.
             * @default warning
             * @enum {string}
             */
            log_level_network?: "debug" | "info" | "warning" | "error" | "critical";
            /**
             * Use Memory Db
             * @description Use in-memory database. Useful for development.
             * @default false
             */
            use_memory_db?: boolean;
            /**
             * Dev Reload
             * @description Automatically reload when Python sources are changed. Does not reload node definitions.
             * @default false
             */
            dev_reload?: boolean;
            /**
             * Profile Graphs
             * @description Enable graph profiling using `cProfile`.
             * @default false
             */
            profile_graphs?: boolean;
            /**
             * Profile Prefix
             * @description An optional prefix for profile output files.
             */
            profile_prefix?: string | null;
            /**
             * Profiles Dir
             * Format: path
             * @description Path to profiles output directory.
             * @default profiles
             */
            profiles_dir?: string;
            /**
             * Max Cache Ram Gb
             * @description The maximum amount of CPU RAM to use for model caching in GB. If unset, the limit will be configured based on the available RAM. In most cases, it is recommended to leave this unset.
             */
            max_cache_ram_gb?: number | null;
            /**
             * Max Cache Vram Gb
             * @description The amount of VRAM to use for model caching in GB. If unset, the limit will be configured based on the available VRAM and the device_working_mem_gb. In most cases, it is recommended to leave this unset.
             */
            max_cache_vram_gb?: number | null;
            /**
             * Log Memory Usage
             * @description If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.
             * @default false
             */
            log_memory_usage?: boolean;
            /**
             * Device Working Mem Gb
             * @description The amount of working memory to keep available on the compute device (in GB). Has no effect if running on CPU. If you are experiencing OOM errors, try increasing this value.
             * @default 3
             */
            device_working_mem_gb?: number;
            /**
             * Enable Partial Loading
             * @description Enable partial loading of models. This enables models to run with reduced VRAM requirements (at the cost of slower speed) by streaming the model from RAM to VRAM as its used. In some edge cases, partial loading can cause models to run more slowly if they were previously being fully loaded into VRAM.
             * @default false
             */
            enable_partial_loading?: boolean;
            /**
             * Keep Ram Copy Of Weights
             * @description Whether to keep a full RAM copy of a model's weights when the model is loaded in VRAM. Keeping a RAM copy increases average RAM usage, but speeds up model switching and LoRA patching (assuming there is sufficient RAM). Set this to False if RAM pressure is consistently high.
             * @default true
             */
            keep_ram_copy_of_weights?: boolean;
            /**
             * Ram
             * @description DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_ram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.
             */
            ram?: number | null;
            /**
             * Vram
             * @description DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_vram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.
             */
            vram?: number | null;
            /**
             * Lazy Offload
             * @description DEPRECATED: This setting is no longer used. Lazy-offloading is enabled by default. This config setting will be removed once the new model cache behavior is stable.
             * @default true
             */
            lazy_offload?: boolean;
            /**
             * Pytorch Cuda Alloc Conf
             * @description Configure the Torch CUDA memory allocator. This will impact peak reserved VRAM usage and performance. Setting to "backend:cudaMallocAsync" works well on many systems. The optimal configuration is highly dependent on the system configuration (device type, VRAM, CUDA driver version, etc.), so must be tuned experimentally.
             */
            pytorch_cuda_alloc_conf?: string | null;
            /**
             * Device
             * @description Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.
             * @default auto
             * @enum {string}
             */
            device?: "auto" | "cpu" | "cuda" | "cuda:1" | "mps";
            /**
             * Precision
             * @description Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.
             * @default auto
             * @enum {string}
             */
            precision?: "auto" | "float16" | "bfloat16" | "float32";
            /**
             * Sequential Guidance
             * @description Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.
             * @default false
             */
            sequential_guidance?: boolean;
            /**
             * Attention Type
             * @description Attention type.
             * @default auto
             * @enum {string}
             */
            attention_type?: "auto" | "normal" | "xformers" | "sliced" | "torch-sdp";
            /**
             * Attention Slice Size
             * @description Slice size, valid when attention_type=="sliced".
             * @default auto
             * @enum {unknown}
             */
            attention_slice_size?: "auto" | "balanced" | "max" | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
            /**
             * Force Tiled Decode
             * @description Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).
             * @default false
             */
            force_tiled_decode?: boolean;
            /**
             * Pil Compress Level
             * @description The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.
             * @default 1
             */
            pil_compress_level?: number;
            /**
             * Max Queue Size
             * @description Maximum number of items in the session queue.
             * @default 10000
             */
            max_queue_size?: number;
            /**
             * Clear Queue On Startup
             * @description Empties session queue on startup.
             * @default false
             */
            clear_queue_on_startup?: boolean;
            /**
             * Allow Nodes
             * @description List of nodes to allow. Omit to allow all.
             */
            allow_nodes?: string[] | null;
            /**
             * Deny Nodes
             * @description List of nodes to deny. Omit to deny none.
             */
            deny_nodes?: string[] | null;
            /**
             * Node Cache Size
             * @description How many cached nodes to keep in memory.
             * @default 512
             */
            node_cache_size?: number;
            /**
             * Hashing Algorithm
             * @description Model hashing algorthim for model installs. 'blake3_multi' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.
             * @default blake3_single
             * @enum {string}
             */
            hashing_algorithm?: "blake3_multi" | "blake3_single" | "random" | "md5" | "sha1" | "sha224" | "sha256" | "sha384" | "sha512" | "blake2b" | "blake2s" | "sha3_224" | "sha3_256" | "sha3_384" | "sha3_512" | "shake_128" | "shake_256";
            /**
             * Remote Api Tokens
             * @description List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.
             */
            remote_api_tokens?: components["schemas"]["URLRegexTokenPair"][] | null;
            /**
             * Scan Models On Startup
             * @description Scan the models directory on startup, registering orphaned models. This is typically only used in conjunction with `use_memory_db` for testing purposes.
             * @default false
             */
            scan_models_on_startup?: boolean;
        };
        /**
         * InvokeAIAppConfigWithSetFields
         * @description InvokeAI App Config with model fields set
         */
        InvokeAIAppConfigWithSetFields: {
            /**
             * Set Fields
             * @description The set fields
             */
            set_fields: string[];
            /** @description The InvokeAI App Config */
            config: components["schemas"]["InvokeAIAppConfig"];
        };
        /**
         * Adjust Image Hue Plus
         * @description Adjusts the Hue of an image by rotating it in the selected color space. Originally created by @dwringer
         */
        InvokeAdjustImageHuePlusInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to adjust
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Space
             * @description Color space in which to rotate hue by polar coords (*: non-invertible)
             * @default HSV / HSL / RGB
             * @enum {string}
             */
            space?: "HSV / HSL / RGB" | "Okhsl" | "Okhsv" | "*Oklch / Oklab" | "*LCh / CIELab" | "*UPLab (w/CIELab_to_UPLab.icc)";
            /**
             * Degrees
             * @description Degrees by which to rotate image hue
             * @default 0
             */
            degrees?: number;
            /**
             * Preserve Lightness
             * @description Whether to preserve CIELAB lightness values
             * @default false
             */
            preserve_lightness?: boolean;
            /**
             * Ok Adaptive Gamut
             * @description Higher preserves chroma at the expense of lightness (Oklab)
             * @default 0.05
             */
            ok_adaptive_gamut?: number;
            /**
             * Ok High Precision
             * @description Use more steps in computing gamut (Oklab/Okhsv/Okhsl)
             * @default true
             */
            ok_high_precision?: boolean;
            /**
             * type
             * @default invokeai_img_hue_adjust_plus
             * @constant
             */
            type: "invokeai_img_hue_adjust_plus";
        };
        /**
         * Equivalent Achromatic Lightness
         * @description Calculate Equivalent Achromatic Lightness from image. Originally created by @dwringer
         */
        InvokeEquivalentAchromaticLightnessInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Image from which to get channel
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default invokeai_ealightness
             * @constant
             */
            type: "invokeai_ealightness";
        };
        /**
         * Image Layer Blend
         * @description Blend two images together, with optional opacity, mask, and blend modes. Originally created by @dwringer
         */
        InvokeImageBlendInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The top image to blend
             * @default null
             */
            layer_upper?: components["schemas"]["ImageField"];
            /**
             * Blend Mode
             * @description Available blend modes
             * @default Normal
             * @enum {string}
             */
            blend_mode?: "Normal" | "Lighten Only" | "Darken Only" | "Lighten Only (EAL)" | "Darken Only (EAL)" | "Hue" | "Saturation" | "Color" | "Luminosity" | "Linear Dodge (Add)" | "Subtract" | "Multiply" | "Divide" | "Screen" | "Overlay" | "Linear Burn" | "Difference" | "Hard Light" | "Soft Light" | "Vivid Light" | "Linear Light" | "Color Burn" | "Color Dodge";
            /**
             * Opacity
             * @description Desired opacity of the upper layer
             * @default 1
             */
            opacity?: number;
            /**
             * @description Optional mask, used to restrict areas from blending
             * @default null
             */
            mask?: components["schemas"]["ImageField"] | null;
            /**
             * Fit To Width
             * @description Scale upper layer to fit base width
             * @default false
             */
            fit_to_width?: boolean;
            /**
             * Fit To Height
             * @description Scale upper layer to fit base height
             * @default true
             */
            fit_to_height?: boolean;
            /**
             * @description The bottom image to blend
             * @default null
             */
            layer_base?: components["schemas"]["ImageField"];
            /**
             * Color Space
             * @description Available color spaces for blend computations
             * @default RGB
             * @enum {string}
             */
            color_space?: "RGB" | "Linear RGB" | "HSL (RGB)" | "HSV (RGB)" | "Okhsl" | "Okhsv" | "Oklch (Oklab)" | "LCh (CIELab)";
            /**
             * Adaptive Gamut
             * @description Adaptive gamut clipping (0=off). Higher prioritizes chroma over lightness
             * @default 0
             */
            adaptive_gamut?: number;
            /**
             * High Precision
             * @description Use more steps in computing gamut when possible
             * @default true
             */
            high_precision?: boolean;
            /**
             * type
             * @default invokeai_img_blend
             * @constant
             */
            type: "invokeai_img_blend";
        };
        /**
         * Image Compositor
         * @description Removes backdrop from subject image then overlays subject on background image. Originally created by @dwringer
         */
        InvokeImageCompositorInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Image of the subject on a plain monochrome background
             * @default null
             */
            image_subject?: components["schemas"]["ImageField"];
            /**
             * @description Image of a background scene
             * @default null
             */
            image_background?: components["schemas"]["ImageField"];
            /**
             * Chroma Key
             * @description Can be empty for corner flood select, or CSS-3 color or tuple
             * @default
             */
            chroma_key?: string;
            /**
             * Threshold
             * @description Subject isolation flood-fill threshold
             * @default 50
             */
            threshold?: number;
            /**
             * Fill X
             * @description Scale base subject image to fit background width
             * @default false
             */
            fill_x?: boolean;
            /**
             * Fill Y
             * @description Scale base subject image to fit background height
             * @default true
             */
            fill_y?: boolean;
            /**
             * X Offset
             * @description x-offset for the subject
             * @default 0
             */
            x_offset?: number;
            /**
             * Y Offset
             * @description y-offset for the subject
             * @default 0
             */
            y_offset?: number;
            /**
             * type
             * @default invokeai_img_composite
             * @constant
             */
            type: "invokeai_img_composite";
        };
        /**
         * Image Dilate or Erode
         * @description Dilate (expand) or erode (contract) an image. Originally created by @dwringer
         */
        InvokeImageDilateOrErodeInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image from which to create a mask
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Lightness Only
             * @description If true, only applies to image lightness (CIELa*b*)
             * @default false
             */
            lightness_only?: boolean;
            /**
             * Radius W
             * @description Width (in pixels) by which to dilate(expand) or erode (contract) the image
             * @default 4
             */
            radius_w?: number;
            /**
             * Radius H
             * @description Height (in pixels) by which to dilate(expand) or erode (contract) the image
             * @default 4
             */
            radius_h?: number;
            /**
             * Mode
             * @description How to operate on the image
             * @default Dilate
             * @enum {string}
             */
            mode?: "Dilate" | "Erode";
            /**
             * type
             * @default invokeai_img_dilate_erode
             * @constant
             */
            type: "invokeai_img_dilate_erode";
        };
        /**
         * Enhance Image
         * @description Applies processing from PIL's ImageEnhance module. Originally created by @dwringer
         */
        InvokeImageEnhanceInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image for which to apply processing
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Invert
             * @description Whether to invert the image colors
             * @default false
             */
            invert?: boolean;
            /**
             * Color
             * @description Color enhancement factor
             * @default 1
             */
            color?: number;
            /**
             * Contrast
             * @description Contrast enhancement factor
             * @default 1
             */
            contrast?: number;
            /**
             * Brightness
             * @description Brightness enhancement factor
             * @default 1
             */
            brightness?: number;
            /**
             * Sharpness
             * @description Sharpness enhancement factor
             * @default 1
             */
            sharpness?: number;
            /**
             * type
             * @default invokeai_img_enhance
             * @constant
             */
            type: "invokeai_img_enhance";
        };
        /**
         * Image Value Thresholds
         * @description Clip image to pure black/white past specified thresholds. Originally created by @dwringer
         */
        InvokeImageValueThresholdsInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image from which to create a mask
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Invert Output
             * @description Make light areas dark and vice versa
             * @default false
             */
            invert_output?: boolean;
            /**
             * Renormalize Values
             * @description Rescale remaining values from minimum to maximum
             * @default false
             */
            renormalize_values?: boolean;
            /**
             * Lightness Only
             * @description If true, only applies to image lightness (CIELa*b*)
             * @default false
             */
            lightness_only?: boolean;
            /**
             * Threshold Upper
             * @description Threshold above which will be set to full value
             * @default 0.5
             */
            threshold_upper?: number;
            /**
             * Threshold Lower
             * @description Threshold below which will be set to minimum value
             * @default 0.5
             */
            threshold_lower?: number;
            /**
             * type
             * @default invokeai_img_val_thresholds
             * @constant
             */
            type: "invokeai_img_val_thresholds";
        };
        /**
         * IterateInvocation
         * @description Iterates over a list of items
         */
        IterateInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The list of items to iterate over
             * @default []
             */
            collection?: unknown[];
            /**
             * Index
             * @description The index, will be provided on executed iterators
             * @default 0
             */
            index?: number;
            /**
             * type
             * @default iterate
             * @constant
             */
            type: "iterate";
        };
        /**
         * IterateInvocationOutput
         * @description Used to connect iteration outputs. Will be expanded to a specific output.
         */
        IterateInvocationOutput: {
            /**
             * Collection Item
             * @description The item being iterated over
             */
            item: unknown;
            /**
             * Index
             * @description The index of the item
             */
            index: number;
            /**
             * Total
             * @description The total number of items
             */
            total: number;
            /**
             * type
             * @default iterate_output
             * @constant
             */
            type: "iterate_output";
        };
        JsonValue: unknown;
        /**
         * LaMa Infill
         * @description Infills transparent areas of an image using the LaMa model
         */
        LaMaInfillInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default infill_lama
             * @constant
             */
            type: "infill_lama";
        };
        /**
         * Latents Collection Primitive
         * @description A collection of latents tensor primitive values
         */
        LatentsCollectionInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The collection of latents tensors
             * @default null
             */
            collection?: components["schemas"]["LatentsField"][];
            /**
             * type
             * @default latents_collection
             * @constant
             */
            type: "latents_collection";
        };
        /**
         * LatentsCollectionOutput
         * @description Base class for nodes that output a collection of latents tensors
         */
        LatentsCollectionOutput: {
            /**
             * Collection
             * @description Latents tensor
             */
            collection: components["schemas"]["LatentsField"][];
            /**
             * type
             * @default latents_collection_output
             * @constant
             */
            type: "latents_collection_output";
        };
        /**
         * LatentsField
         * @description A latents tensor primitive field
         */
        LatentsField: {
            /**
             * Latents Name
             * @description The name of the latents
             */
            latents_name: string;
            /**
             * Seed
             * @description Seed used to generate this latents
             * @default null
             */
            seed?: number | null;
        };
        /**
         * Latents Primitive
         * @description A latents tensor primitive value
         */
        LatentsInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * type
             * @default latents
             * @constant
             */
            type: "latents";
        };
        /**
         * LatentsMetaOutput
         * @description Latents + metadata
         */
        LatentsMetaOutput: {
            /** @description Metadata Dict */
            metadata: components["schemas"]["MetadataField"];
            /**
             * type
             * @default latents_meta_output
             * @constant
             */
            type: "latents_meta_output";
            /** @description Latents tensor */
            latents: components["schemas"]["LatentsField"];
            /**
             * Width
             * @description Width of output (px)
             */
            width: number;
            /**
             * Height
             * @description Height of output (px)
             */
            height: number;
        };
        /**
         * LatentsOutput
         * @description Base class for nodes that output a single latents tensor
         */
        LatentsOutput: {
            /** @description Latents tensor */
            latents: components["schemas"]["LatentsField"];
            /**
             * Width
             * @description Width of output (px)
             */
            width: number;
            /**
             * Height
             * @description Height of output (px)
             */
            height: number;
            /**
             * type
             * @default latents_output
             * @constant
             */
            type: "latents_output";
        };
        /**
         * Latents to Image - SD1.5, SDXL
         * @description Generates an image from latents.
         */
        LatentsToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * Tiled
             * @description Processing using overlapping tiles (reduce memory consumption)
             * @default false
             */
            tiled?: boolean;
            /**
             * Tile Size
             * @description The tile size for VAE tiling in pixels (image space). If set to 0, the default tile size for the model will be used. Larger tile sizes generally produce better results at the cost of higher memory usage.
             * @default 0
             */
            tile_size?: number;
            /**
             * Fp32
             * @description Whether or not to use full float32 precision
             * @default false
             */
            fp32?: boolean;
            /**
             * type
             * @default l2i
             * @constant
             */
            type: "l2i";
        };
        /**
         * Lineart Anime Edge Detection
         * @description Geneartes an edge map using the Lineart model.
         */
        LineartAnimeEdgeDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default lineart_anime_edge_detection
             * @constant
             */
            type: "lineart_anime_edge_detection";
        };
        /**
         * Lineart Edge Detection
         * @description Generates an edge map using the Lineart model.
         */
        LineartEdgeDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Coarse
             * @description Whether to use coarse mode
             * @default false
             */
            coarse?: boolean;
            /**
             * type
             * @default lineart_edge_detection
             * @constant
             */
            type: "lineart_edge_detection";
        };
        /**
         * LlavaOnevisionConfig
         * @description Model config for Llava Onevision models.
         */
        LlavaOnevisionConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default llava_onevision
             * @constant
             */
            type: "llava_onevision";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
        };
        /**
         * LLaVA OneVision VLLM
         * @description Run a LLaVA OneVision VLLM model.
         */
        LlavaOnevisionVllmInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Images
             * @description Input image.
             * @default null
             */
            images?: (components["schemas"]["ImageField"][] | components["schemas"]["ImageField"]) | null;
            /**
             * Prompt
             * @description Input text prompt.
             * @default
             */
            prompt?: string;
            /**
             * LLaVA Model Type
             * @description The VLLM model to use
             * @default null
             */
            vllm_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default llava_onevision_vllm
             * @constant
             */
            type: "llava_onevision_vllm";
        };
        /**
         * Apply LoRA Collection - SD1.5
         * @description Applies a collection of LoRAs to the provided UNet and CLIP models.
         */
        LoRACollectionLoader: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * LoRAs
             * @description LoRA models and weights. May be a single LoRA or collection.
             * @default null
             */
            loras?: components["schemas"]["LoRAField"] | components["schemas"]["LoRAField"][] | null;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default lora_collection_loader
             * @constant
             */
            type: "lora_collection_loader";
        };
        /**
         * LoRADiffusersConfig
         * @description Model config for LoRA/Diffusers models.
         */
        LoRADiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default lora
             * @constant
             */
            type: "lora";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
        };
        /** LoRAField */
        LoRAField: {
            /** @description Info to load lora model */
            lora: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description Weight to apply to lora model
             */
            weight: number;
        };
        /**
         * Apply LoRA - SD1.5
         * @description Apply selected lora to unet and text_encoder.
         */
        LoRALoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * LoRA
             * @description LoRA model to load
             * @default null
             */
            lora?: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight at which the LoRA is applied to each model
             * @default 0.75
             */
            weight?: number;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default lora_loader
             * @constant
             */
            type: "lora_loader";
        };
        /**
         * LoRALoaderOutput
         * @description Model loader output
         */
        LoRALoaderOutput: {
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet: components["schemas"]["UNetField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default lora_loader_output
             * @constant
             */
            type: "lora_loader_output";
        };
        /**
         * LoRALyCORISConfig
         * @description Model config for LoRA/Lycoris models.
         */
        LoRALyCORISConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default lora
             * @constant
             */
            type: "lora";
            /**
             * Format
             * @default lycoris
             * @constant
             */
            format: "lycoris";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
        };
        /**
         * LoRAMetadataField
         * @description LoRA Metadata Field
         */
        LoRAMetadataField: {
            /** @description LoRA model to load */
            model: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight at which the LoRA is applied to each model
             */
            weight: number;
        };
        /**
         * Select LoRA
         * @description Selects a LoRA model and weight.
         */
        LoRASelectorInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * LoRA
             * @description LoRA model to load
             * @default null
             */
            lora?: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight at which the LoRA is applied to each model
             * @default 0.75
             */
            weight?: number;
            /**
             * type
             * @default lora_selector
             * @constant
             */
            type: "lora_selector";
        };
        /**
         * LoRASelectorOutput
         * @description Model loader output
         */
        LoRASelectorOutput: {
            /**
             * LoRA
             * @description LoRA model and weight
             */
            lora: components["schemas"]["LoRAField"];
            /**
             * type
             * @default lora_selector_output
             * @constant
             */
            type: "lora_selector_output";
        };
        /**
         * LocalModelSource
         * @description A local file or directory path.
         */
        LocalModelSource: {
            /** Path */
            path: string;
            /**
             * Inplace
             * @default false
             */
            inplace?: boolean | null;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "local";
        };
        /**
         * LogLevel
         * @enum {integer}
         */
        LogLevel: 0 | 10 | 20 | 30 | 40 | 50;
        /** MDControlListOutput */
        MDControlListOutput: {
            /**
             * ControlNet-List
             * @description ControlNet(s) to apply
             */
            control_list: components["schemas"]["ControlField"] | components["schemas"]["ControlField"][] | null;
            /**
             * type
             * @default md_control_list_output
             * @constant
             */
            type: "md_control_list_output";
        };
        /** MDIPAdapterListOutput */
        MDIPAdapterListOutput: {
            /**
             * IP-Adapter-List
             * @description IP-Adapter to apply
             */
            ip_adapter_list: components["schemas"]["IPAdapterField"] | components["schemas"]["IPAdapterField"][] | null;
            /**
             * type
             * @default md_ip_adapter_list_output
             * @constant
             */
            type: "md_ip_adapter_list_output";
        };
        /** MDT2IAdapterListOutput */
        MDT2IAdapterListOutput: {
            /**
             * T2I Adapter-List
             * @description T2I-Adapter(s) to apply
             */
            t2i_adapter_list: components["schemas"]["T2IAdapterField"] | components["schemas"]["T2IAdapterField"][] | null;
            /**
             * type
             * @default md_ip_adapters_output
             * @constant
             */
            type: "md_ip_adapters_output";
        };
        /**
         * MLSD Detection
         * @description Generates an line segment map using MLSD.
         */
        MLSDDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Score Threshold
             * @description The threshold used to score points when determining line segments
             * @default 0.1
             */
            score_threshold?: number;
            /**
             * Distance Threshold
             * @description Threshold for including a line segment - lines shorter than this distance will be discarded
             * @default 20
             */
            distance_threshold?: number;
            /**
             * type
             * @default mlsd_detection
             * @constant
             */
            type: "mlsd_detection";
        };
        /**
         * MainBnbQuantized4bCheckpointConfig
         * @description Model config for main checkpoint models.
         */
        MainBnbQuantized4bCheckpointConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default main
             * @constant
             */
            type: "main";
            /**
             * Format
             * @default bnb_quantized_nf4b
             * @constant
             */
            format: "bnb_quantized_nf4b";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["MainModelDefaultSettings"] | null;
            /**
             * Variant
             * @default normal
             */
            variant?: components["schemas"]["ModelVariantType"] | components["schemas"]["ClipVariantType"] | null;
            /**
             * Config Path
             * @description path to the checkpoint model config file
             */
            config_path: string;
            /**
             * Converted At
             * @description When this model was last converted to diffusers
             */
            converted_at?: number | null;
            /** @default epsilon */
            prediction_type?: components["schemas"]["SchedulerPredictionType"];
            /**
             * Upcast Attention
             * @default false
             */
            upcast_attention?: boolean;
        };
        /**
         * MainCheckpointConfig
         * @description Model config for main checkpoint models.
         */
        MainCheckpointConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default main
             * @constant
             */
            type: "main";
            /**
             * Format
             * @description Format of the provided checkpoint model
             * @default checkpoint
             * @enum {string}
             */
            format: "checkpoint" | "bnb_quantized_nf4b" | "gguf_quantized";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["MainModelDefaultSettings"] | null;
            /**
             * Variant
             * @default normal
             */
            variant?: components["schemas"]["ModelVariantType"] | components["schemas"]["ClipVariantType"] | null;
            /**
             * Config Path
             * @description path to the checkpoint model config file
             */
            config_path: string;
            /**
             * Converted At
             * @description When this model was last converted to diffusers
             */
            converted_at?: number | null;
            /** @default epsilon */
            prediction_type?: components["schemas"]["SchedulerPredictionType"];
            /**
             * Upcast Attention
             * @default false
             */
            upcast_attention?: boolean;
        };
        /**
         * MainDiffusersConfig
         * @description Model config for main diffusers models.
         */
        MainDiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default main
             * @constant
             */
            type: "main";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["MainModelDefaultSettings"] | null;
            /**
             * Variant
             * @default normal
             */
            variant?: components["schemas"]["ModelVariantType"] | components["schemas"]["ClipVariantType"] | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
        };
        /**
         * MainGGUFCheckpointConfig
         * @description Model config for main checkpoint models.
         */
        MainGGUFCheckpointConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default main
             * @constant
             */
            type: "main";
            /**
             * Format
             * @default gguf_quantized
             * @constant
             */
            format: "gguf_quantized";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["MainModelDefaultSettings"] | null;
            /**
             * Variant
             * @default normal
             */
            variant?: components["schemas"]["ModelVariantType"] | components["schemas"]["ClipVariantType"] | null;
            /**
             * Config Path
             * @description path to the checkpoint model config file
             */
            config_path: string;
            /**
             * Converted At
             * @description When this model was last converted to diffusers
             */
            converted_at?: number | null;
            /** @default epsilon */
            prediction_type?: components["schemas"]["SchedulerPredictionType"];
            /**
             * Upcast Attention
             * @default false
             */
            upcast_attention?: boolean;
        };
        /** MainModelDefaultSettings */
        MainModelDefaultSettings: {
            /**
             * Vae
             * @description Default VAE for this model (model key)
             */
            vae?: string | null;
            /**
             * Vae Precision
             * @description Default VAE precision for this model
             */
            vae_precision?: ("fp16" | "fp32") | null;
            /**
             * Scheduler
             * @description Default scheduler for this model
             */
            scheduler?: ("ddim" | "ddpm" | "deis" | "deis_k" | "lms" | "lms_k" | "pndm" | "heun" | "heun_k" | "euler" | "euler_k" | "euler_a" | "kdpm_2" | "kdpm_2_k" | "kdpm_2_a" | "kdpm_2_a_k" | "dpmpp_2s" | "dpmpp_2s_k" | "dpmpp_2m" | "dpmpp_2m_k" | "dpmpp_2m_sde" | "dpmpp_2m_sde_k" | "dpmpp_3m" | "dpmpp_3m_k" | "dpmpp_sde" | "dpmpp_sde_k" | "unipc" | "unipc_k" | "lcm" | "tcd") | null;
            /**
             * Steps
             * @description Default number of steps for this model
             */
            steps?: number | null;
            /**
             * Cfg Scale
             * @description Default CFG Scale for this model
             */
            cfg_scale?: number | null;
            /**
             * Cfg Rescale Multiplier
             * @description Default CFG Rescale Multiplier for this model
             */
            cfg_rescale_multiplier?: number | null;
            /**
             * Width
             * @description Default width for this model
             */
            width?: number | null;
            /**
             * Height
             * @description Default height for this model
             */
            height?: number | null;
            /**
             * Guidance
             * @description Default Guidance for this model
             */
            guidance?: number | null;
        };
        /**
         * Main Model - SD1.5
         * @description Loads a main model, outputting its submodels.
         */
        MainModelLoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Main model (UNet, VAE, CLIP) to load
             * @default null
             */
            model?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default main_model_loader
             * @constant
             */
            type: "main_model_loader";
        };
        /**
         * Combine Masks
         * @description Combine two masks together by multiplying them using `PIL.ImageChops.multiply()`.
         */
        MaskCombineInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The first mask to combine
             * @default null
             */
            mask1?: components["schemas"]["ImageField"];
            /**
             * @description The second image to combine
             * @default null
             */
            mask2?: components["schemas"]["ImageField"];
            /**
             * type
             * @default mask_combine
             * @constant
             */
            type: "mask_combine";
        };
        /**
         * Mask Edge
         * @description Applies an edge mask to an image
         */
        MaskEdgeInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to apply the mask to
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Edge Size
             * @description The size of the edge
             * @default null
             */
            edge_size?: number;
            /**
             * Edge Blur
             * @description The amount of blur on the edge
             * @default null
             */
            edge_blur?: number;
            /**
             * Low Threshold
             * @description First threshold for the hysteresis procedure in Canny edge detection
             * @default null
             */
            low_threshold?: number;
            /**
             * High Threshold
             * @description Second threshold for the hysteresis procedure in Canny edge detection
             * @default null
             */
            high_threshold?: number;
            /**
             * type
             * @default mask_edge
             * @constant
             */
            type: "mask_edge";
        };
        /**
         * Mask from Alpha
         * @description Extracts the alpha channel of an image as a mask.
         */
        MaskFromAlphaInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to create the mask from
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Invert
             * @description Whether or not to invert the mask
             * @default false
             */
            invert?: boolean;
            /**
             * type
             * @default tomask
             * @constant
             */
            type: "tomask";
        };
        /**
         * Mask from Segmented Image
         * @description Generate a mask for a particular color in an ID Map
         */
        MaskFromIDInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to create the mask from
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description ID color to mask
             * @default null
             */
            color?: components["schemas"]["ColorField"];
            /**
             * Threshold
             * @description Threshold for color detection
             * @default 100
             */
            threshold?: number;
            /**
             * Invert
             * @description Whether or not to invert the mask
             * @default false
             */
            invert?: boolean;
            /**
             * type
             * @default mask_from_id
             * @constant
             */
            type: "mask_from_id";
        };
        /**
         * MaskOutput
         * @description A torch mask tensor.
         */
        MaskOutput: {
            /** @description The mask. */
            mask: components["schemas"]["TensorField"];
            /**
             * Width
             * @description The width of the mask in pixels.
             */
            width: number;
            /**
             * Height
             * @description The height of the mask in pixels.
             */
            height: number;
            /**
             * type
             * @default mask_output
             * @constant
             */
            type: "mask_output";
        };
        /**
         * Tensor Mask to Image
         * @description Convert a mask tensor to an image.
         */
        MaskTensorToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The mask tensor to convert.
             * @default null
             */
            mask?: components["schemas"]["TensorField"];
            /**
             * type
             * @default tensor_mask_to_image
             * @constant
             */
            type: "tensor_mask_to_image";
        };
        /**
         * MediaPipe Face Detection
         * @description Detects faces using MediaPipe.
         */
        MediaPipeFaceDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Max Faces
             * @description Maximum number of faces to detect
             * @default 1
             */
            max_faces?: number;
            /**
             * Min Confidence
             * @description Minimum confidence for face detection
             * @default 0.5
             */
            min_confidence?: number;
            /**
             * type
             * @default mediapipe_face_detection
             * @constant
             */
            type: "mediapipe_face_detection";
        };
        /**
         * Metadata Merge
         * @description Merged a collection of MetadataDict into a single MetadataDict.
         */
        MergeMetadataInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description Collection of Metadata
             * @default null
             */
            collection?: components["schemas"]["MetadataField"][];
            /**
             * type
             * @default merge_metadata
             * @constant
             */
            type: "merge_metadata";
        };
        /**
         * Merge Tiles to Image
         * @description Merge multiple tile images into a single image.
         */
        MergeTilesToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Tiles With Images
             * @description A list of tile images with tile properties.
             * @default null
             */
            tiles_with_images?: components["schemas"]["TileWithImage"][];
            /**
             * Blend Mode
             * @description blending type Linear or Seam
             * @default Seam
             * @enum {string}
             */
            blend_mode?: "Linear" | "Seam";
            /**
             * Blend Amount
             * @description The amount to blend adjacent tiles in pixels. Must be <= the amount of overlap between adjacent tiles.
             * @default 32
             */
            blend_amount?: number;
            /**
             * type
             * @default merge_tiles_to_image
             * @constant
             */
            type: "merge_tiles_to_image";
        };
        /**
         * MetadataField
         * @description Pydantic model for metadata with custom root of type dict[str, Any].
         *     Metadata is stored without a strict schema.
         */
        MetadataField: Record<string, unknown>;
        /**
         * Metadata Field Extractor
         * @description Extracts the text value from an image's metadata given a key.
         *     Raises an error if the image has no metadata or if the value is not a string (nesting not permitted).
         */
        MetadataFieldExtractorInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to extract metadata from
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Key
             * @description The key in the image's metadata to extract the value from
             * @default null
             */
            key?: string;
            /**
             * type
             * @default metadata_field_extractor
             * @constant
             */
            type: "metadata_field_extractor";
        };
        /**
         * Metadata From Image
         * @description Used to create a core metadata item then Add/Update it to the provided metadata
         */
        MetadataFromImageInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default metadata_from_image
             * @constant
             */
            type: "metadata_from_image";
        };
        /**
         * Metadata
         * @description Takes a MetadataItem or collection of MetadataItems and outputs a MetadataDict.
         */
        MetadataInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Items
             * @description A single metadata item or collection of metadata items
             * @default null
             */
            items?: components["schemas"]["MetadataItemField"][] | components["schemas"]["MetadataItemField"];
            /**
             * type
             * @default metadata
             * @constant
             */
            type: "metadata";
        };
        /** MetadataItemField */
        MetadataItemField: {
            /**
             * Label
             * @description Label for this metadata item
             */
            label: string;
            /**
             * Value
             * @description The value for this metadata item (may be any type)
             */
            value: unknown;
        };
        /**
         * Metadata Item
         * @description Used to create an arbitrary metadata item. Provide "label" and make a connection to "value" to store that data as the value.
         */
        MetadataItemInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default null
             */
            label?: string;
            /**
             * Value
             * @description The value for this metadata item (may be any type)
             * @default null
             */
            value?: unknown;
            /**
             * type
             * @default metadata_item
             * @constant
             */
            type: "metadata_item";
        };
        /**
         * Metadata Item Linked
         * @description Used to Create/Add/Update a value into a metadata label
         */
        MetadataItemLinkedInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default * CUSTOM LABEL *
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "positive_prompt" | "positive_style_prompt" | "negative_prompt" | "negative_style_prompt" | "width" | "height" | "seed" | "cfg_scale" | "cfg_rescale_multiplier" | "steps" | "scheduler" | "clip_skip" | "model" | "vae" | "seamless_x" | "seamless_y" | "guidance" | "cfg_scale_start_step" | "cfg_scale_end_step";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * Value
             * @description The value for this metadata item (may be any type)
             * @default null
             */
            value?: unknown;
            /**
             * type
             * @default metadata_item_linked
             * @constant
             */
            type: "metadata_item_linked";
        };
        /**
         * MetadataItemOutput
         * @description Metadata Item Output
         */
        MetadataItemOutput: {
            /** @description Metadata Item */
            item: components["schemas"]["MetadataItemField"];
            /**
             * type
             * @default metadata_item_output
             * @constant
             */
            type: "metadata_item_output";
        };
        /** MetadataOutput */
        MetadataOutput: {
            /** @description Metadata Dict */
            metadata: components["schemas"]["MetadataField"];
            /**
             * type
             * @default metadata_output
             * @constant
             */
            type: "metadata_output";
        };
        /**
         * Metadata To Bool
         * @description Extracts a Boolean value of a label from metadata
         */
        MetadataToBoolInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default * CUSTOM LABEL *
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "seamless_x" | "seamless_y";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * Default Value
             * @description The default bool to use if not found in the metadata
             * @default null
             */
            default_value?: boolean;
            /**
             * type
             * @default metadata_to_bool
             * @constant
             */
            type: "metadata_to_bool";
        };
        /**
         * Metadata To ControlNets
         * @description Extracts a Controlnets value of a label from metadata
         */
        MetadataToControlnetsInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * ControlNet-List
             * @default null
             */
            control_list?: components["schemas"]["ControlField"] | components["schemas"]["ControlField"][] | null;
            /**
             * type
             * @default metadata_to_controlnets
             * @constant
             */
            type: "metadata_to_controlnets";
        };
        /**
         * Metadata To Float
         * @description Extracts a Float value of a label from metadata
         */
        MetadataToFloatInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default * CUSTOM LABEL *
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "cfg_scale" | "cfg_rescale_multiplier" | "guidance";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * Default Value
             * @description The default float to use if not found in the metadata
             * @default null
             */
            default_value?: number;
            /**
             * type
             * @default metadata_to_float
             * @constant
             */
            type: "metadata_to_float";
        };
        /**
         * Metadata To IP-Adapters
         * @description Extracts a IP-Adapters value of a label from metadata
         */
        MetadataToIPAdaptersInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * IP-Adapter-List
             * @description IP-Adapter to apply
             * @default null
             */
            ip_adapter_list?: components["schemas"]["IPAdapterField"] | components["schemas"]["IPAdapterField"][] | null;
            /**
             * type
             * @default metadata_to_ip_adapters
             * @constant
             */
            type: "metadata_to_ip_adapters";
        };
        /**
         * Metadata To Integer
         * @description Extracts an integer value of a label from metadata
         */
        MetadataToIntegerInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default * CUSTOM LABEL *
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "width" | "height" | "seed" | "steps" | "clip_skip" | "cfg_scale_start_step" | "cfg_scale_end_step";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * Default Value
             * @description The default integer to use if not found in the metadata
             * @default null
             */
            default_value?: number;
            /**
             * type
             * @default metadata_to_integer
             * @constant
             */
            type: "metadata_to_integer";
        };
        /**
         * Metadata To LoRA Collection
         * @description Extracts Lora(s) from metadata into a collection
         */
        MetadataToLorasCollectionInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default loras
             */
            custom_label?: string;
            /**
             * LoRAs
             * @description LoRA models and weights. May be a single LoRA or collection.
             * @default []
             */
            loras?: components["schemas"]["LoRAField"] | components["schemas"]["LoRAField"][] | null;
            /**
             * type
             * @default metadata_to_lora_collection
             * @constant
             */
            type: "metadata_to_lora_collection";
        };
        /**
         * MetadataToLorasCollectionOutput
         * @description Model loader output
         */
        MetadataToLorasCollectionOutput: {
            /**
             * LoRAs
             * @description Collection of LoRA model and weights
             */
            lora: components["schemas"]["LoRAField"][];
            /**
             * type
             * @default metadata_to_lora_collection_output
             * @constant
             */
            type: "metadata_to_lora_collection_output";
        };
        /**
         * Metadata To LoRAs
         * @description Extracts a Loras value of a label from metadata
         */
        MetadataToLorasInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default metadata_to_loras
             * @constant
             */
            type: "metadata_to_loras";
        };
        /**
         * Metadata To Model
         * @description Extracts a Model value of a label from metadata
         */
        MetadataToModelInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default model
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "model";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * @description The default model to use if not found in the metadata
             * @default null
             */
            default_value?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default metadata_to_model
             * @constant
             */
            type: "metadata_to_model";
        };
        /**
         * MetadataToModelOutput
         * @description String to main model output
         */
        MetadataToModelOutput: {
            /**
             * Model
             * @description Main model (UNet, VAE, CLIP) to load
             */
            model: components["schemas"]["ModelIdentifierField"];
            /**
             * Name
             * @description Model Name
             */
            name: string;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             */
            unet: components["schemas"]["UNetField"];
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip: components["schemas"]["CLIPField"];
            /**
             * type
             * @default metadata_to_model_output
             * @constant
             */
            type: "metadata_to_model_output";
        };
        /**
         * Metadata To SDXL LoRAs
         * @description Extracts a SDXL Loras value of a label from metadata
         */
        MetadataToSDXLLorasInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * CLIP 1
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip2?: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default metadata_to_sdlx_loras
             * @constant
             */
            type: "metadata_to_sdlx_loras";
        };
        /**
         * Metadata To SDXL Model
         * @description Extracts a SDXL Model value of a label from metadata
         */
        MetadataToSDXLModelInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default model
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "model";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * @description The default SDXL Model to use if not found in the metadata
             * @default null
             */
            default_value?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default metadata_to_sdxl_model
             * @constant
             */
            type: "metadata_to_sdxl_model";
        };
        /**
         * MetadataToSDXLModelOutput
         * @description String to SDXL main model output
         */
        MetadataToSDXLModelOutput: {
            /**
             * Model
             * @description Main model (UNet, VAE, CLIP) to load
             */
            model: components["schemas"]["ModelIdentifierField"];
            /**
             * Name
             * @description Model Name
             */
            name: string;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             */
            unet: components["schemas"]["UNetField"];
            /**
             * CLIP 1
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip: components["schemas"]["CLIPField"];
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip2: components["schemas"]["CLIPField"];
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * type
             * @default metadata_to_sdxl_model_output
             * @constant
             */
            type: "metadata_to_sdxl_model_output";
        };
        /**
         * Metadata To Scheduler
         * @description Extracts a Scheduler value of a label from metadata
         */
        MetadataToSchedulerInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default scheduler
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "scheduler";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * Default Value
             * @description The default scheduler to use if not found in the metadata
             * @default euler
             * @enum {string}
             */
            default_value?: "ddim" | "ddpm" | "deis" | "deis_k" | "lms" | "lms_k" | "pndm" | "heun" | "heun_k" | "euler" | "euler_k" | "euler_a" | "kdpm_2" | "kdpm_2_k" | "kdpm_2_a" | "kdpm_2_a_k" | "dpmpp_2s" | "dpmpp_2s_k" | "dpmpp_2m" | "dpmpp_2m_k" | "dpmpp_2m_sde" | "dpmpp_2m_sde_k" | "dpmpp_3m" | "dpmpp_3m_k" | "dpmpp_sde" | "dpmpp_sde_k" | "unipc" | "unipc_k" | "lcm" | "tcd";
            /**
             * type
             * @default metadata_to_scheduler
             * @constant
             */
            type: "metadata_to_scheduler";
        };
        /**
         * Metadata To String
         * @description Extracts a string value of a label from metadata
         */
        MetadataToStringInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default * CUSTOM LABEL *
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "positive_prompt" | "positive_style_prompt" | "negative_prompt" | "negative_style_prompt";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * Default Value
             * @description The default string to use if not found in the metadata
             * @default null
             */
            default_value?: string;
            /**
             * type
             * @default metadata_to_string
             * @constant
             */
            type: "metadata_to_string";
        };
        /**
         * Metadata To T2I-Adapters
         * @description Extracts a T2I-Adapters value of a label from metadata
         */
        MetadataToT2IAdaptersInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * T2I-Adapter
             * @description IP-Adapter to apply
             * @default null
             */
            t2i_adapter_list?: components["schemas"]["T2IAdapterField"] | components["schemas"]["T2IAdapterField"][] | null;
            /**
             * type
             * @default metadata_to_t2i_adapters
             * @constant
             */
            type: "metadata_to_t2i_adapters";
        };
        /**
         * Metadata To VAE
         * @description Extracts a VAE value of a label from metadata
         */
        MetadataToVAEInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Label
             * @description Label for this metadata item
             * @default vae
             * @enum {string}
             */
            label?: "* CUSTOM LABEL *" | "vae";
            /**
             * Custom Label
             * @description Label for this metadata item
             * @default null
             */
            custom_label?: string | null;
            /**
             * @description The default VAE to use if not found in the metadata
             * @default null
             */
            default_value?: components["schemas"]["VAEField"];
            /**
             * type
             * @default metadata_to_vae
             * @constant
             */
            type: "metadata_to_vae";
        };
        /**
         * ModelFormat
         * @description Storage format of model.
         * @enum {string}
         */
        ModelFormat: "diffusers" | "checkpoint" | "lycoris" | "onnx" | "olive" | "embedding_file" | "embedding_folder" | "invokeai" | "t5_encoder" | "bnb_quantized_int8b" | "bnb_quantized_nf4b" | "gguf_quantized";
        /** ModelIdentifierField */
        ModelIdentifierField: {
            /**
             * Key
             * @description The model's unique key
             */
            key: string;
            /**
             * Hash
             * @description The model's BLAKE3 hash
             */
            hash: string;
            /**
             * Name
             * @description The model's name
             */
            name: string;
            /** @description The model's base model type */
            base: components["schemas"]["BaseModelType"];
            /** @description The model's type */
            type: components["schemas"]["ModelType"];
            /**
             * @description The submodel to load, if this is a main model
             * @default null
             */
            submodel_type?: components["schemas"]["SubModelType"] | null;
        };
        /**
         * Any Model
         * @description Selects any model, outputting it its identifier. Be careful with this one! The identifier will be accepted as
         *     input for any model, even if the model types don't match. If you connect this to a mismatched input, you'll get an
         *     error.
         */
        ModelIdentifierInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Model
             * @description The model to select
             * @default null
             */
            model?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default model_identifier
             * @constant
             */
            type: "model_identifier";
        };
        /**
         * ModelIdentifierOutput
         * @description Model identifier output
         */
        ModelIdentifierOutput: {
            /**
             * Model
             * @description Model identifier
             */
            model: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default model_identifier_output
             * @constant
             */
            type: "model_identifier_output";
        };
        /**
         * ModelInstallCancelledEvent
         * @description Event model for model_install_cancelled
         */
        ModelInstallCancelledEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Id
             * @description The ID of the install job
             */
            id: number;
            /**
             * Source
             * @description Source of the model; local path, repo_id or url
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
        };
        /**
         * ModelInstallCompleteEvent
         * @description Event model for model_install_complete
         */
        ModelInstallCompleteEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Id
             * @description The ID of the install job
             */
            id: number;
            /**
             * Source
             * @description Source of the model; local path, repo_id or url
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
            /**
             * Key
             * @description Model config record key
             */
            key: string;
            /**
             * Total Bytes
             * @description Size of the model (may be None for installation of a local path)
             */
            total_bytes: number | null;
        };
        /**
         * ModelInstallDownloadProgressEvent
         * @description Event model for model_install_download_progress
         */
        ModelInstallDownloadProgressEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Id
             * @description The ID of the install job
             */
            id: number;
            /**
             * Source
             * @description Source of the model; local path, repo_id or url
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
            /**
             * Local Path
             * @description Where model is downloading to
             */
            local_path: string;
            /**
             * Bytes
             * @description Number of bytes downloaded so far
             */
            bytes: number;
            /**
             * Total Bytes
             * @description Total size of download, including all files
             */
            total_bytes: number;
            /**
             * Parts
             * @description Progress of downloading URLs that comprise the model, if any
             */
            parts: {
                [key: string]: number | string;
            }[];
        };
        /**
         * ModelInstallDownloadStartedEvent
         * @description Event model for model_install_download_started
         */
        ModelInstallDownloadStartedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Id
             * @description The ID of the install job
             */
            id: number;
            /**
             * Source
             * @description Source of the model; local path, repo_id or url
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
            /**
             * Local Path
             * @description Where model is downloading to
             */
            local_path: string;
            /**
             * Bytes
             * @description Number of bytes downloaded so far
             */
            bytes: number;
            /**
             * Total Bytes
             * @description Total size of download, including all files
             */
            total_bytes: number;
            /**
             * Parts
             * @description Progress of downloading URLs that comprise the model, if any
             */
            parts: {
                [key: string]: number | string;
            }[];
        };
        /**
         * ModelInstallDownloadsCompleteEvent
         * @description Emitted once when an install job becomes active.
         */
        ModelInstallDownloadsCompleteEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Id
             * @description The ID of the install job
             */
            id: number;
            /**
             * Source
             * @description Source of the model; local path, repo_id or url
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
        };
        /**
         * ModelInstallErrorEvent
         * @description Event model for model_install_error
         */
        ModelInstallErrorEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Id
             * @description The ID of the install job
             */
            id: number;
            /**
             * Source
             * @description Source of the model; local path, repo_id or url
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
            /**
             * Error Type
             * @description The name of the exception
             */
            error_type: string;
            /**
             * Error
             * @description A text description of the exception
             */
            error: string;
        };
        /**
         * ModelInstallJob
         * @description Object that tracks the current status of an install request.
         */
        ModelInstallJob: {
            /**
             * Id
             * @description Unique ID for this job
             */
            id: number;
            /**
             * @description Current status of install process
             * @default waiting
             */
            status?: components["schemas"]["InstallStatus"];
            /**
             * Error Reason
             * @description Information about why the job failed
             */
            error_reason?: string | null;
            /** @description Configuration information (e.g. 'description') to apply to model. */
            config_in?: components["schemas"]["ModelRecordChanges"];
            /**
             * Config Out
             * @description After successful installation, this will hold the configuration object.
             */
            config_out?: (components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"]) | null;
            /**
             * Inplace
             * @description Leave model in its current location; otherwise install under models directory
             * @default false
             */
            inplace?: boolean;
            /**
             * Source
             * @description Source (URL, repo_id, or local path) of model
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
            /**
             * Local Path
             * Format: path
             * @description Path to locally-downloaded model; may be the same as the source
             */
            local_path: string;
            /**
             * Bytes
             * @description For a remote model, the number of bytes downloaded so far (may not be available)
             * @default 0
             */
            bytes?: number;
            /**
             * Total Bytes
             * @description Total size of the model to be installed
             * @default 0
             */
            total_bytes?: number;
            /**
             * Source Metadata
             * @description Metadata provided by the model source
             */
            source_metadata?: (components["schemas"]["BaseMetadata"] | components["schemas"]["HuggingFaceMetadata"]) | null;
            /**
             * Download Parts
             * @description Download jobs contributing to this install
             */
            download_parts?: components["schemas"]["DownloadJob"][];
            /**
             * Error
             * @description On an error condition, this field will contain the text of the exception
             */
            error?: string | null;
            /**
             * Error Traceback
             * @description On an error condition, this field will contain the exception traceback
             */
            error_traceback?: string | null;
        };
        /**
         * ModelInstallStartedEvent
         * @description Event model for model_install_started
         */
        ModelInstallStartedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Id
             * @description The ID of the install job
             */
            id: number;
            /**
             * Source
             * @description Source of the model; local path, repo_id or url
             */
            source: components["schemas"]["LocalModelSource"] | components["schemas"]["HFModelSource"] | components["schemas"]["URLModelSource"];
        };
        /**
         * ModelLoadCompleteEvent
         * @description Event model for model_load_complete
         */
        ModelLoadCompleteEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Config
             * @description The model's config
             */
            config: components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"];
            /**
             * @description The submodel type, if any
             * @default null
             */
            submodel_type: components["schemas"]["SubModelType"] | null;
        };
        /**
         * ModelLoadStartedEvent
         * @description Event model for model_load_started
         */
        ModelLoadStartedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Config
             * @description The model's config
             */
            config: components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"];
            /**
             * @description The submodel type, if any
             * @default null
             */
            submodel_type: components["schemas"]["SubModelType"] | null;
        };
        /**
         * ModelLoaderOutput
         * @description Model loader output
         */
        ModelLoaderOutput: {
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * type
             * @default model_loader_output
             * @constant
             */
            type: "model_loader_output";
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip: components["schemas"]["CLIPField"];
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             */
            unet: components["schemas"]["UNetField"];
        };
        /**
         * ModelRecordChanges
         * @description A set of changes to apply to a model.
         */
        ModelRecordChanges: {
            /**
             * Source
             * @description original source of the model
             */
            source?: string | null;
            /** @description type of model source */
            source_type?: components["schemas"]["ModelSourceType"] | null;
            /**
             * Source Api Response
             * @description metadata from remote source
             */
            source_api_response?: string | null;
            /**
             * Name
             * @description Name of the model.
             */
            name?: string | null;
            /**
             * Path
             * @description Path to the model.
             */
            path?: string | null;
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /** @description The base model. */
            base?: components["schemas"]["BaseModelType"] | null;
            /** @description Type of model */
            type?: components["schemas"]["ModelType"] | null;
            /**
             * Key
             * @description Database ID for this model
             */
            key?: string | null;
            /**
             * Hash
             * @description hash of model file
             */
            hash?: string | null;
            /**
             * File Size
             * @description Size of model file
             */
            file_size?: number | null;
            /**
             * Format
             * @description format of model file
             */
            format?: string | null;
            /**
             * Trigger Phrases
             * @description Set of trigger phrases for this model
             */
            trigger_phrases?: string[] | null;
            /**
             * Default Settings
             * @description Default settings for this model
             */
            default_settings?: components["schemas"]["MainModelDefaultSettings"] | components["schemas"]["ControlAdapterDefaultSettings"] | null;
            /**
             * Variant
             * @description The variant of the model.
             */
            variant?: components["schemas"]["ModelVariantType"] | components["schemas"]["ClipVariantType"] | null;
            /** @description The prediction type of the model. */
            prediction_type?: components["schemas"]["SchedulerPredictionType"] | null;
            /**
             * Upcast Attention
             * @description Whether to upcast attention.
             */
            upcast_attention?: boolean | null;
            /**
             * Config Path
             * @description Path to config file for model
             */
            config_path?: string | null;
        };
        /**
         * ModelRepoVariant
         * @description Various hugging face variants on the diffusers format.
         * @enum {string}
         */
        ModelRepoVariant: "" | "fp16" | "fp32" | "onnx" | "openvino" | "flax";
        /**
         * ModelSourceType
         * @description Model source type.
         * @enum {string}
         */
        ModelSourceType: "path" | "url" | "hf_repo_id";
        /**
         * ModelType
         * @description Model type.
         * @enum {string}
         */
        ModelType: "onnx" | "main" | "vae" | "lora" | "control_lora" | "controlnet" | "embedding" | "ip_adapter" | "clip_vision" | "clip_embed" | "t2i_adapter" | "t5_encoder" | "spandrel_image_to_image" | "siglip" | "flux_redux" | "llava_onevision";
        /**
         * ModelVariantType
         * @description Variant type.
         * @enum {string}
         */
        ModelVariantType: "normal" | "inpaint" | "depth";
        /**
         * ModelsList
         * @description Return list of configs.
         */
        ModelsList: {
            /** Models */
            models: (components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"])[];
        };
        /**
         * Multiply Integers
         * @description Multiplies two numbers
         */
        MultiplyInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * A
             * @description The first number
             * @default 0
             */
            a?: number;
            /**
             * B
             * @description The second number
             * @default 0
             */
            b?: number;
            /**
             * type
             * @default mul
             * @constant
             */
            type: "mul";
        };
        /** NodeFieldValue */
        NodeFieldValue: {
            /**
             * Node Path
             * @description The node into which this batch data item will be substituted.
             */
            node_path: string;
            /**
             * Field Name
             * @description The field into which this batch data item will be substituted.
             */
            field_name: string;
            /**
             * Value
             * @description The value to substitute into the node/field.
             */
            value: string | number | components["schemas"]["ImageField"];
        };
        /**
         * Create Latent Noise
         * @description Generates latent noise.
         */
        NoiseInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Seed
             * @description Seed for random number generation
             * @default 0
             */
            seed?: number;
            /**
             * Width
             * @description Width of output (px)
             * @default 512
             */
            width?: number;
            /**
             * Height
             * @description Height of output (px)
             * @default 512
             */
            height?: number;
            /**
             * Use Cpu
             * @description Use CPU for noise generation (for reproducible results across platforms)
             * @default true
             */
            use_cpu?: boolean;
            /**
             * type
             * @default noise
             * @constant
             */
            type: "noise";
        };
        /**
         * NoiseOutput
         * @description Invocation noise output
         */
        NoiseOutput: {
            /** @description Noise tensor */
            noise: components["schemas"]["LatentsField"];
            /**
             * Width
             * @description Width of output (px)
             */
            width: number;
            /**
             * Height
             * @description Height of output (px)
             */
            height: number;
            /**
             * type
             * @default noise_output
             * @constant
             */
            type: "noise_output";
        };
        /**
         * Normal Map
         * @description Generates a normal map.
         */
        NormalMapInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default normal_map
             * @constant
             */
            type: "normal_map";
        };
        /** OffsetPaginatedResults[BoardDTO] */
        OffsetPaginatedResults_BoardDTO_: {
            /**
             * Limit
             * @description Limit of items to get
             */
            limit: number;
            /**
             * Offset
             * @description Offset from which to retrieve items
             */
            offset: number;
            /**
             * Total
             * @description Total number of items in result
             */
            total: number;
            /**
             * Items
             * @description Items
             */
            items: components["schemas"]["BoardDTO"][];
        };
        /** OffsetPaginatedResults[ImageDTO] */
        OffsetPaginatedResults_ImageDTO_: {
            /**
             * Limit
             * @description Limit of items to get
             */
            limit: number;
            /**
             * Offset
             * @description Offset from which to retrieve items
             */
            offset: number;
            /**
             * Total
             * @description Total number of items in result
             */
            total: number;
            /**
             * Items
             * @description Items
             */
            items: components["schemas"]["ImageDTO"][];
        };
        /**
         * OutputFieldJSONSchemaExtra
         * @description Extra attributes to be added to input fields and their OpenAPI schema. Used by the workflow editor
         *     during schema parsing and UI rendering.
         */
        OutputFieldJSONSchemaExtra: {
            field_kind: components["schemas"]["FieldKind"];
            /** Ui Hidden */
            ui_hidden: boolean;
            ui_type: components["schemas"]["UIType"] | null;
            /** Ui Order */
            ui_order: number | null;
        };
        /** PaginatedResults[WorkflowRecordListItemWithThumbnailDTO] */
        PaginatedResults_WorkflowRecordListItemWithThumbnailDTO_: {
            /**
             * Page
             * @description Current Page
             */
            page: number;
            /**
             * Pages
             * @description Total number of pages
             */
            pages: number;
            /**
             * Per Page
             * @description Number of items per page
             */
            per_page: number;
            /**
             * Total
             * @description Total number of items in result
             */
            total: number;
            /**
             * Items
             * @description Items
             */
            items: components["schemas"]["WorkflowRecordListItemWithThumbnailDTO"][];
        };
        /**
         * Pair Tile with Image
         * @description Pair an image with its tile properties.
         */
        PairTileImageInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The tile image.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description The tile properties.
             * @default null
             */
            tile?: components["schemas"]["Tile"];
            /**
             * type
             * @default pair_tile_image
             * @constant
             */
            type: "pair_tile_image";
        };
        /** PairTileImageOutput */
        PairTileImageOutput: {
            /** @description A tile description with its corresponding image. */
            tile_with_image: components["schemas"]["TileWithImage"];
            /**
             * type
             * @default pair_tile_image_output
             * @constant
             */
            type: "pair_tile_image_output";
        };
        /**
         * Paste Image into Bounding Box
         * @description Paste the source image into the target image at the given bounding box.
         *
         *     The source image must be the same size as the bounding box, and the bounding box must fit within the target image.
         */
        PasteImageIntoBoundingBoxInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to paste
             * @default null
             */
            source_image?: components["schemas"]["ImageField"];
            /**
             * @description The image to paste into
             * @default null
             */
            target_image?: components["schemas"]["ImageField"];
            /**
             * @description The bounding box to paste the image into
             * @default null
             */
            bounding_box?: components["schemas"]["BoundingBoxField"];
            /**
             * type
             * @default paste_image_into_bounding_box
             * @constant
             */
            type: "paste_image_into_bounding_box";
        };
        /**
         * PiDiNet Edge Detection
         * @description Generates an edge map using PiDiNet.
         */
        PiDiNetEdgeDetectionInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Quantize Edges
             * @description Whether or not to use safe mode
             * @default false
             */
            quantize_edges?: boolean;
            /**
             * Scribble
             * @description Whether or not to use scribble mode
             * @default false
             */
            scribble?: boolean;
            /**
             * type
             * @default pidi_edge_detection
             * @constant
             */
            type: "pidi_edge_detection";
        };
        /** PresetData */
        PresetData: {
            /**
             * Positive Prompt
             * @description Positive prompt
             */
            positive_prompt: string;
            /**
             * Negative Prompt
             * @description Negative prompt
             */
            negative_prompt: string;
        };
        /**
         * PresetType
         * @enum {string}
         */
        PresetType: "user" | "default" | "project";
        /**
         * ProgressImage
         * @description The progress image sent intermittently during processing
         */
        ProgressImage: {
            /**
             * Width
             * @description The effective width of the image in pixels
             */
            width: number;
            /**
             * Height
             * @description The effective height of the image in pixels
             */
            height: number;
            /**
             * Dataurl
             * @description The image data as a b64 data URL
             */
            dataURL: string;
        };
        /**
         * Prompts from File
         * @description Loads prompts from a text file
         */
        PromptsFromFileInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * File Path
             * @description Path to prompt text file
             * @default null
             */
            file_path?: string;
            /**
             * Pre Prompt
             * @description String to prepend to each prompt
             * @default null
             */
            pre_prompt?: string | null;
            /**
             * Post Prompt
             * @description String to append to each prompt
             * @default null
             */
            post_prompt?: string | null;
            /**
             * Start Line
             * @description Line in the file to start start from
             * @default 1
             */
            start_line?: number;
            /**
             * Max Prompts
             * @description Max lines to read from file (0=all)
             * @default 1
             */
            max_prompts?: number;
            /**
             * type
             * @default prompt_from_file
             * @constant
             */
            type: "prompt_from_file";
        };
        /**
         * PruneResult
         * @description Result of pruning the session queue
         */
        PruneResult: {
            /**
             * Deleted
             * @description Number of queue items deleted
             */
            deleted: number;
        };
        /**
         * QueueClearedEvent
         * @description Event model for queue_cleared
         */
        QueueClearedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
        };
        /**
         * QueueItemStatusChangedEvent
         * @description Event model for queue_item_status_changed
         */
        QueueItemStatusChangedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Item Id
             * @description The ID of the queue item
             */
            item_id: number;
            /**
             * Batch Id
             * @description The ID of the queue batch
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of the queue item
             * @default null
             */
            origin: string | null;
            /**
             * Destination
             * @description The destination of the queue item
             * @default null
             */
            destination: string | null;
            /**
             * Status
             * @description The new status of the queue item
             * @enum {string}
             */
            status: "pending" | "in_progress" | "completed" | "failed" | "canceled";
            /**
             * Error Type
             * @description The error type, if any
             * @default null
             */
            error_type: string | null;
            /**
             * Error Message
             * @description The error message, if any
             * @default null
             */
            error_message: string | null;
            /**
             * Error Traceback
             * @description The error traceback, if any
             * @default null
             */
            error_traceback: string | null;
            /**
             * Created At
             * @description The timestamp when the queue item was created
             * @default null
             */
            created_at: string | null;
            /**
             * Updated At
             * @description The timestamp when the queue item was last updated
             * @default null
             */
            updated_at: string | null;
            /**
             * Started At
             * @description The timestamp when the queue item was started
             * @default null
             */
            started_at: string | null;
            /**
             * Completed At
             * @description The timestamp when the queue item was completed
             * @default null
             */
            completed_at: string | null;
            /** @description The status of the batch */
            batch_status: components["schemas"]["BatchStatus"];
            /** @description The status of the queue */
            queue_status: components["schemas"]["SessionQueueStatus"];
            /**
             * Session Id
             * @description The ID of the session (aka graph execution state)
             */
            session_id: string;
        };
        /**
         * QueueItemsRetriedEvent
         * @description Event model for queue_items_retried
         */
        QueueItemsRetriedEvent: {
            /**
             * Timestamp
             * @description The timestamp of the event
             */
            timestamp: number;
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Retried Item Ids
             * @description The IDs of the queue items that were retried
             */
            retried_item_ids: number[];
        };
        /**
         * Random Float
         * @description Outputs a single random float
         */
        RandomFloatInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default false
             */
            use_cache?: boolean;
            /**
             * Low
             * @description The inclusive low value
             * @default 0
             */
            low?: number;
            /**
             * High
             * @description The exclusive high value
             * @default 1
             */
            high?: number;
            /**
             * Decimals
             * @description The number of decimal places to round to
             * @default 2
             */
            decimals?: number;
            /**
             * type
             * @default rand_float
             * @constant
             */
            type: "rand_float";
        };
        /**
         * Random Integer
         * @description Outputs a single random integer.
         */
        RandomIntInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default false
             */
            use_cache?: boolean;
            /**
             * Low
             * @description The inclusive low value
             * @default 0
             */
            low?: number;
            /**
             * High
             * @description The exclusive high value
             * @default 2147483647
             */
            high?: number;
            /**
             * type
             * @default rand_int
             * @constant
             */
            type: "rand_int";
        };
        /**
         * Random Range
         * @description Creates a collection of random numbers
         */
        RandomRangeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default false
             */
            use_cache?: boolean;
            /**
             * Low
             * @description The inclusive low value
             * @default 0
             */
            low?: number;
            /**
             * High
             * @description The exclusive high value
             * @default 2147483647
             */
            high?: number;
            /**
             * Size
             * @description The number of values to generate
             * @default 1
             */
            size?: number;
            /**
             * Seed
             * @description The seed for the RNG (omit for random)
             * @default 0
             */
            seed?: number;
            /**
             * type
             * @default random_range
             * @constant
             */
            type: "random_range";
        };
        /**
         * Integer Range
         * @description Creates a range of numbers from start to stop with step
         */
        RangeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Start
             * @description The start of the range
             * @default 0
             */
            start?: number;
            /**
             * Stop
             * @description The stop of the range
             * @default 10
             */
            stop?: number;
            /**
             * Step
             * @description The step of the range
             * @default 1
             */
            step?: number;
            /**
             * type
             * @default range
             * @constant
             */
            type: "range";
        };
        /**
         * Integer Range of Size
         * @description Creates a range from start to start + (size * step) incremented by step
         */
        RangeOfSizeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Start
             * @description The start of the range
             * @default 0
             */
            start?: number;
            /**
             * Size
             * @description The number of values
             * @default 1
             */
            size?: number;
            /**
             * Step
             * @description The step of the range
             * @default 1
             */
            step?: number;
            /**
             * type
             * @default range_of_size
             * @constant
             */
            type: "range_of_size";
        };
        /**
         * Create Rectangle Mask
         * @description Create a rectangular mask.
         */
        RectangleMaskInvocation: {
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Width
             * @description The width of the entire mask.
             * @default null
             */
            width?: number;
            /**
             * Height
             * @description The height of the entire mask.
             * @default null
             */
            height?: number;
            /**
             * X Left
             * @description The left x-coordinate of the rectangular masked region (inclusive).
             * @default null
             */
            x_left?: number;
            /**
             * Y Top
             * @description The top y-coordinate of the rectangular masked region (inclusive).
             * @default null
             */
            y_top?: number;
            /**
             * Rectangle Width
             * @description The width of the rectangular masked region.
             * @default null
             */
            rectangle_width?: number;
            /**
             * Rectangle Height
             * @description The height of the rectangular masked region.
             * @default null
             */
            rectangle_height?: number;
            /**
             * type
             * @default rectangle_mask
             * @constant
             */
            type: "rectangle_mask";
        };
        /**
         * RemoteModelFile
         * @description Information about a downloadable file that forms part of a model.
         */
        RemoteModelFile: {
            /**
             * Url
             * Format: uri
             * @description The url to download this model file
             */
            url: string;
            /**
             * Path
             * Format: path
             * @description The path to the file, relative to the model root
             */
            path: string;
            /**
             * Size
             * @description The size of this file, in bytes
             * @default 0
             */
            size?: number | null;
            /**
             * Sha256
             * @description SHA256 hash of this model (not always available)
             */
            sha256?: string | null;
        };
        /** RemoveImagesFromBoardResult */
        RemoveImagesFromBoardResult: {
            /**
             * Removed Image Names
             * @description The image names that were removed from their board
             */
            removed_image_names: string[];
        };
        /**
         * Resize Latents
         * @description Resizes latents to explicit width/height (in pixels). Provided dimensions are floor-divided by 8.
         */
        ResizeLatentsInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * Width
             * @description Width of output (px)
             * @default null
             */
            width?: number;
            /**
             * Height
             * @description Width of output (px)
             * @default null
             */
            height?: number;
            /**
             * Mode
             * @description Interpolation mode
             * @default bilinear
             * @enum {string}
             */
            mode?: "nearest" | "linear" | "bilinear" | "bicubic" | "trilinear" | "area" | "nearest-exact";
            /**
             * Antialias
             * @description Whether or not to apply antialiasing (bilinear or bicubic only)
             * @default false
             */
            antialias?: boolean;
            /**
             * type
             * @default lresize
             * @constant
             */
            type: "lresize";
        };
        /**
         * ResourceOrigin
         * @description The origin of a resource (eg image).
         *
         *     - INTERNAL: The resource was created by the application.
         *     - EXTERNAL: The resource was not created by the application.
         *     This may be a user-initiated upload, or an internal application upload (eg Canvas init image).
         * @enum {string}
         */
        ResourceOrigin: "internal" | "external";
        /** RetryItemsResult */
        RetryItemsResult: {
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Retried Item Ids
             * @description The IDs of the queue items that were retried
             */
            retried_item_ids: number[];
        };
        /**
         * Round Float
         * @description Rounds a float to a specified number of decimal places.
         */
        RoundInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Value
             * @description The float value
             * @default 0
             */
            value?: number;
            /**
             * Decimals
             * @description The number of decimal places
             * @default 0
             */
            decimals?: number;
            /**
             * type
             * @default round_float
             * @constant
             */
            type: "round_float";
        };
        /** SAMPoint */
        SAMPoint: {
            /**
             * X
             * @description The x-coordinate of the point
             */
            x: number;
            /**
             * Y
             * @description The y-coordinate of the point
             */
            y: number;
            /** @description The label of the point */
            label: components["schemas"]["SAMPointLabel"];
        };
        /**
         * SAMPointLabel
         * @enum {integer}
         */
        SAMPointLabel: -1 | 0 | 1;
        /** SAMPointsField */
        SAMPointsField: {
            /**
             * Points
             * @description The points of the object
             */
            points: components["schemas"]["SAMPoint"][];
        };
        /**
         * SD3ConditioningField
         * @description A conditioning tensor primitive value
         */
        SD3ConditioningField: {
            /**
             * Conditioning Name
             * @description The name of conditioning tensor
             */
            conditioning_name: string;
        };
        /**
         * SD3ConditioningOutput
         * @description Base class for nodes that output a single SD3 conditioning tensor
         */
        SD3ConditioningOutput: {
            /** @description Conditioning tensor */
            conditioning: components["schemas"]["SD3ConditioningField"];
            /**
             * type
             * @default sd3_conditioning_output
             * @constant
             */
            type: "sd3_conditioning_output";
        };
        /**
         * Denoise - SD3
         * @description Run denoising process with a SD3 model.
         */
        SD3DenoiseInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"] | null;
            /**
             * @description A mask of the region to apply the denoising process to. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved.
             * @default null
             */
            denoise_mask?: components["schemas"]["DenoiseMaskField"] | null;
            /**
             * Denoising Start
             * @description When to start denoising, expressed a percentage of total steps
             * @default 0
             */
            denoising_start?: number;
            /**
             * Denoising End
             * @description When to stop denoising, expressed a percentage of total steps
             * @default 1
             */
            denoising_end?: number;
            /**
             * Transformer
             * @description SD3 model (MMDiTX) to load
             * @default null
             */
            transformer?: components["schemas"]["TransformerField"];
            /**
             * @description Positive conditioning tensor
             * @default null
             */
            positive_conditioning?: components["schemas"]["SD3ConditioningField"];
            /**
             * @description Negative conditioning tensor
             * @default null
             */
            negative_conditioning?: components["schemas"]["SD3ConditioningField"];
            /**
             * CFG Scale
             * @description Classifier-Free Guidance scale
             * @default 3.5
             */
            cfg_scale?: number | number[];
            /**
             * Width
             * @description Width of the generated image.
             * @default 1024
             */
            width?: number;
            /**
             * Height
             * @description Height of the generated image.
             * @default 1024
             */
            height?: number;
            /**
             * Steps
             * @description Number of steps to run
             * @default 10
             */
            steps?: number;
            /**
             * Seed
             * @description Randomness seed for reproducibility.
             * @default 0
             */
            seed?: number;
            /**
             * type
             * @default sd3_denoise
             * @constant
             */
            type: "sd3_denoise";
        };
        /**
         * Image to Latents - SD3
         * @description Generates latents from an image.
         */
        SD3ImageToLatentsInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to encode
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * type
             * @default sd3_i2l
             * @constant
             */
            type: "sd3_i2l";
        };
        /**
         * Latents to Image - SD3
         * @description Generates an image from latents.
         */
        SD3LatentsToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * @description VAE
             * @default null
             */
            vae?: components["schemas"]["VAEField"];
            /**
             * type
             * @default sd3_l2i
             * @constant
             */
            type: "sd3_l2i";
        };
        /**
         * Prompt - SDXL
         * @description Parse prompt using compel package to conditioning.
         */
        SDXLCompelPromptInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Prompt
             * @description Prompt to be parsed by Compel to create a conditioning tensor
             * @default
             */
            prompt?: string;
            /**
             * Style
             * @description Prompt to be parsed by Compel to create a conditioning tensor
             * @default
             */
            style?: string;
            /**
             * Original Width
             * @default 1024
             */
            original_width?: number;
            /**
             * Original Height
             * @default 1024
             */
            original_height?: number;
            /**
             * Crop Top
             * @default 0
             */
            crop_top?: number;
            /**
             * Crop Left
             * @default 0
             */
            crop_left?: number;
            /**
             * Target Width
             * @default 1024
             */
            target_width?: number;
            /**
             * Target Height
             * @default 1024
             */
            target_height?: number;
            /**
             * CLIP 1
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"];
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip2?: components["schemas"]["CLIPField"];
            /**
             * @description A mask defining the region that this conditioning prompt applies to.
             * @default null
             */
            mask?: components["schemas"]["TensorField"] | null;
            /**
             * type
             * @default sdxl_compel_prompt
             * @constant
             */
            type: "sdxl_compel_prompt";
        };
        /**
         * Apply LoRA Collection - SDXL
         * @description Applies a collection of SDXL LoRAs to the provided UNet and CLIP models.
         */
        SDXLLoRACollectionLoader: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * LoRAs
             * @description LoRA models and weights. May be a single LoRA or collection.
             * @default null
             */
            loras?: components["schemas"]["LoRAField"] | components["schemas"]["LoRAField"][] | null;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * CLIP
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip2?: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default sdxl_lora_collection_loader
             * @constant
             */
            type: "sdxl_lora_collection_loader";
        };
        /**
         * Apply LoRA - SDXL
         * @description Apply selected lora to unet and text_encoder.
         */
        SDXLLoRALoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * LoRA
             * @description LoRA model to load
             * @default null
             */
            lora?: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight at which the LoRA is applied to each model
             * @default 0.75
             */
            weight?: number;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * CLIP 1
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip?: components["schemas"]["CLIPField"] | null;
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip2?: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default sdxl_lora_loader
             * @constant
             */
            type: "sdxl_lora_loader";
        };
        /**
         * SDXLLoRALoaderOutput
         * @description SDXL LoRA Loader Output
         */
        SDXLLoRALoaderOutput: {
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet: components["schemas"]["UNetField"] | null;
            /**
             * CLIP 1
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip: components["schemas"]["CLIPField"] | null;
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip2: components["schemas"]["CLIPField"] | null;
            /**
             * type
             * @default sdxl_lora_loader_output
             * @constant
             */
            type: "sdxl_lora_loader_output";
        };
        /**
         * Main Model - SDXL
         * @description Loads an sdxl base model, outputting its submodels.
         */
        SDXLModelLoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description SDXL Main model (UNet, VAE, CLIP1, CLIP2) to load
             * @default null
             */
            model?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default sdxl_model_loader
             * @constant
             */
            type: "sdxl_model_loader";
        };
        /**
         * SDXLModelLoaderOutput
         * @description SDXL base model loader output
         */
        SDXLModelLoaderOutput: {
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             */
            unet: components["schemas"]["UNetField"];
            /**
             * CLIP 1
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip: components["schemas"]["CLIPField"];
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip2: components["schemas"]["CLIPField"];
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * type
             * @default sdxl_model_loader_output
             * @constant
             */
            type: "sdxl_model_loader_output";
        };
        /**
         * Prompt - SDXL Refiner
         * @description Parse prompt using compel package to conditioning.
         */
        SDXLRefinerCompelPromptInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Style
             * @description Prompt to be parsed by Compel to create a conditioning tensor
             * @default
             */
            style?: string;
            /**
             * Original Width
             * @default 1024
             */
            original_width?: number;
            /**
             * Original Height
             * @default 1024
             */
            original_height?: number;
            /**
             * Crop Top
             * @default 0
             */
            crop_top?: number;
            /**
             * Crop Left
             * @default 0
             */
            crop_left?: number;
            /**
             * Aesthetic Score
             * @description The aesthetic score to apply to the conditioning tensor
             * @default 6
             */
            aesthetic_score?: number;
            /**
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip2?: components["schemas"]["CLIPField"];
            /**
             * type
             * @default sdxl_refiner_compel_prompt
             * @constant
             */
            type: "sdxl_refiner_compel_prompt";
        };
        /**
         * Refiner Model - SDXL
         * @description Loads an sdxl refiner model, outputting its submodels.
         */
        SDXLRefinerModelLoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description SDXL Refiner Main Modde (UNet, VAE, CLIP2) to load
             * @default null
             */
            model?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default sdxl_refiner_model_loader
             * @constant
             */
            type: "sdxl_refiner_model_loader";
        };
        /**
         * SDXLRefinerModelLoaderOutput
         * @description SDXL refiner model loader output
         */
        SDXLRefinerModelLoaderOutput: {
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             */
            unet: components["schemas"]["UNetField"];
            /**
             * CLIP 2
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip2: components["schemas"]["CLIPField"];
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * type
             * @default sdxl_refiner_model_loader_output
             * @constant
             */
            type: "sdxl_refiner_model_loader_output";
        };
        /**
         * SQLiteDirection
         * @enum {string}
         */
        SQLiteDirection: "ASC" | "DESC";
        /**
         * Save Image
         * @description Saves an image. Unlike an image primitive, this invocation stores a copy of the image.
         */
        SaveImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default false
             */
            use_cache?: boolean;
            /**
             * @description The image to process
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default save_image
             * @constant
             */
            type: "save_image";
        };
        /**
         * Scale Latents
         * @description Scales latents by a given factor.
         */
        ScaleLatentsInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"];
            /**
             * Scale Factor
             * @description The factor by which to scale
             * @default null
             */
            scale_factor?: number;
            /**
             * Mode
             * @description Interpolation mode
             * @default bilinear
             * @enum {string}
             */
            mode?: "nearest" | "linear" | "bilinear" | "bicubic" | "trilinear" | "area" | "nearest-exact";
            /**
             * Antialias
             * @description Whether or not to apply antialiasing (bilinear or bicubic only)
             * @default false
             */
            antialias?: boolean;
            /**
             * type
             * @default lscale
             * @constant
             */
            type: "lscale";
        };
        /**
         * Scheduler
         * @description Selects a scheduler.
         */
        SchedulerInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Scheduler
             * @description Scheduler to use during inference
             * @default euler
             * @enum {string}
             */
            scheduler?: "ddim" | "ddpm" | "deis" | "deis_k" | "lms" | "lms_k" | "pndm" | "heun" | "heun_k" | "euler" | "euler_k" | "euler_a" | "kdpm_2" | "kdpm_2_k" | "kdpm_2_a" | "kdpm_2_a_k" | "dpmpp_2s" | "dpmpp_2s_k" | "dpmpp_2m" | "dpmpp_2m_k" | "dpmpp_2m_sde" | "dpmpp_2m_sde_k" | "dpmpp_3m" | "dpmpp_3m_k" | "dpmpp_sde" | "dpmpp_sde_k" | "unipc" | "unipc_k" | "lcm" | "tcd";
            /**
             * type
             * @default scheduler
             * @constant
             */
            type: "scheduler";
        };
        /** SchedulerOutput */
        SchedulerOutput: {
            /**
             * Scheduler
             * @description Scheduler to use during inference
             * @enum {string}
             */
            scheduler: "ddim" | "ddpm" | "deis" | "deis_k" | "lms" | "lms_k" | "pndm" | "heun" | "heun_k" | "euler" | "euler_k" | "euler_a" | "kdpm_2" | "kdpm_2_k" | "kdpm_2_a" | "kdpm_2_a_k" | "dpmpp_2s" | "dpmpp_2s_k" | "dpmpp_2m" | "dpmpp_2m_k" | "dpmpp_2m_sde" | "dpmpp_2m_sde_k" | "dpmpp_3m" | "dpmpp_3m_k" | "dpmpp_sde" | "dpmpp_sde_k" | "unipc" | "unipc_k" | "lcm" | "tcd";
            /**
             * type
             * @default scheduler_output
             * @constant
             */
            type: "scheduler_output";
        };
        /**
         * SchedulerPredictionType
         * @description Scheduler prediction type.
         * @enum {string}
         */
        SchedulerPredictionType: "epsilon" | "v_prediction" | "sample";
        /**
         * Main Model - SD3
         * @description Loads a SD3 base model, outputting its submodels.
         */
        Sd3ModelLoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /** @description SD3 model (MMDiTX) to load */
            model: components["schemas"]["ModelIdentifierField"];
            /**
             * T5 Encoder
             * @description T5 tokenizer and text encoder
             * @default null
             */
            t5_encoder_model?: components["schemas"]["ModelIdentifierField"] | null;
            /**
             * CLIP L Encoder
             * @description CLIP Embed loader
             * @default null
             */
            clip_l_model?: components["schemas"]["ModelIdentifierField"] | null;
            /**
             * CLIP G Encoder
             * @description CLIP-G Embed loader
             * @default null
             */
            clip_g_model?: components["schemas"]["ModelIdentifierField"] | null;
            /**
             * VAE
             * @description VAE model to load
             * @default null
             */
            vae_model?: components["schemas"]["ModelIdentifierField"] | null;
            /**
             * type
             * @default sd3_model_loader
             * @constant
             */
            type: "sd3_model_loader";
        };
        /**
         * Sd3ModelLoaderOutput
         * @description SD3 base model loader output.
         */
        Sd3ModelLoaderOutput: {
            /**
             * Transformer
             * @description Transformer
             */
            transformer: components["schemas"]["TransformerField"];
            /**
             * CLIP L
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip_l: components["schemas"]["CLIPField"];
            /**
             * CLIP G
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             */
            clip_g: components["schemas"]["CLIPField"];
            /**
             * T5 Encoder
             * @description T5 tokenizer and text encoder
             */
            t5_encoder: components["schemas"]["T5EncoderField"];
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * type
             * @default sd3_model_loader_output
             * @constant
             */
            type: "sd3_model_loader_output";
        };
        /**
         * Prompt - SD3
         * @description Encodes and preps a prompt for a SD3 image.
         */
        Sd3TextEncoderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * CLIP L
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip_l?: components["schemas"]["CLIPField"];
            /**
             * CLIP G
             * @description CLIP (tokenizer, text encoder, LoRAs) and skipped layer count
             * @default null
             */
            clip_g?: components["schemas"]["CLIPField"];
            /**
             * T5Encoder
             * @description T5 tokenizer and text encoder
             * @default null
             */
            t5_encoder?: components["schemas"]["T5EncoderField"] | null;
            /**
             * Prompt
             * @description Text prompt to encode.
             * @default null
             */
            prompt?: string;
            /**
             * type
             * @default sd3_text_encoder
             * @constant
             */
            type: "sd3_text_encoder";
        };
        /**
         * Apply Seamless - SD1.5, SDXL
         * @description Applies the seamless transformation to the Model UNet and VAE.
         */
        SeamlessModeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"] | null;
            /**
             * VAE
             * @description VAE model to load
             * @default null
             */
            vae?: components["schemas"]["VAEField"] | null;
            /**
             * Seamless Y
             * @description Specify whether Y axis is seamless
             * @default true
             */
            seamless_y?: boolean;
            /**
             * Seamless X
             * @description Specify whether X axis is seamless
             * @default true
             */
            seamless_x?: boolean;
            /**
             * type
             * @default seamless
             * @constant
             */
            type: "seamless";
        };
        /**
         * SeamlessModeOutput
         * @description Modified Seamless Model output
         */
        SeamlessModeOutput: {
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet: components["schemas"]["UNetField"] | null;
            /**
             * VAE
             * @description VAE
             * @default null
             */
            vae: components["schemas"]["VAEField"] | null;
            /**
             * type
             * @default seamless_output
             * @constant
             */
            type: "seamless_output";
        };
        /**
         * Segment Anything
         * @description Runs a Segment Anything Model.
         */
        SegmentAnythingInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Model
             * @description The Segment Anything model to use.
             * @default null
             * @enum {string}
             */
            model?: "segment-anything-base" | "segment-anything-large" | "segment-anything-huge";
            /**
             * @description The image to segment.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Bounding Boxes
             * @description The bounding boxes to prompt the SAM model with.
             * @default null
             */
            bounding_boxes?: components["schemas"]["BoundingBoxField"][] | null;
            /**
             * Point Lists
             * @description The list of point lists to prompt the SAM model with. Each list of points represents a single object.
             * @default null
             */
            point_lists?: components["schemas"]["SAMPointsField"][] | null;
            /**
             * Apply Polygon Refinement
             * @description Whether to apply polygon refinement to the masks. This will smooth the edges of the masks slightly and ensure that each mask consists of a single closed polygon (before merging).
             * @default true
             */
            apply_polygon_refinement?: boolean;
            /**
             * Mask Filter
             * @description The filtering to apply to the detected masks before merging them into a final output.
             * @default all
             * @enum {string}
             */
            mask_filter?: "all" | "largest" | "highest_box_score";
            /**
             * type
             * @default segment_anything
             * @constant
             */
            type: "segment_anything";
        };
        /** SessionProcessorStatus */
        SessionProcessorStatus: {
            /**
             * Is Started
             * @description Whether the session processor is started
             */
            is_started: boolean;
            /**
             * Is Processing
             * @description Whether a session is being processed
             */
            is_processing: boolean;
        };
        /**
         * SessionQueueAndProcessorStatus
         * @description The overall status of session queue and processor
         */
        SessionQueueAndProcessorStatus: {
            queue: components["schemas"]["SessionQueueStatus"];
            processor: components["schemas"]["SessionProcessorStatus"];
        };
        /** SessionQueueCountsByDestination */
        SessionQueueCountsByDestination: {
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Destination
             * @description The destination of queue items included in this status
             */
            destination: string;
            /**
             * Pending
             * @description Number of queue items with status 'pending' for the destination
             */
            pending: number;
            /**
             * In Progress
             * @description Number of queue items with status 'in_progress' for the destination
             */
            in_progress: number;
            /**
             * Completed
             * @description Number of queue items with status 'complete' for the destination
             */
            completed: number;
            /**
             * Failed
             * @description Number of queue items with status 'error' for the destination
             */
            failed: number;
            /**
             * Canceled
             * @description Number of queue items with status 'canceled' for the destination
             */
            canceled: number;
            /**
             * Total
             * @description Total number of queue items for the destination
             */
            total: number;
        };
        /** SessionQueueItem */
        SessionQueueItem: {
            /**
             * Item Id
             * @description The identifier of the session queue item
             */
            item_id: number;
            /**
             * Status
             * @description The status of this queue item
             * @default pending
             * @enum {string}
             */
            status: "pending" | "in_progress" | "completed" | "failed" | "canceled";
            /**
             * Priority
             * @description The priority of this queue item
             * @default 0
             */
            priority: number;
            /**
             * Batch Id
             * @description The ID of the batch associated with this queue item
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of this queue item. This data is used by the frontend to determine how to handle results.
             */
            origin?: string | null;
            /**
             * Destination
             * @description The origin of this queue item. This data is used by the frontend to determine how to handle results
             */
            destination?: string | null;
            /**
             * Session Id
             * @description The ID of the session associated with this queue item. The session doesn't exist in graph_executions until the queue item is executed.
             */
            session_id: string;
            /**
             * Error Type
             * @description The error type if this queue item errored
             */
            error_type?: string | null;
            /**
             * Error Message
             * @description The error message if this queue item errored
             */
            error_message?: string | null;
            /**
             * Error Traceback
             * @description The error traceback if this queue item errored
             */
            error_traceback?: string | null;
            /**
             * Created At
             * @description When this queue item was created
             */
            created_at: string;
            /**
             * Updated At
             * @description When this queue item was updated
             */
            updated_at: string;
            /**
             * Started At
             * @description When this queue item was started
             */
            started_at?: string | null;
            /**
             * Completed At
             * @description When this queue item was completed
             */
            completed_at?: string | null;
            /**
             * Queue Id
             * @description The id of the queue with which this item is associated
             */
            queue_id: string;
            /**
             * Field Values
             * @description The field values that were used for this queue item
             */
            field_values?: components["schemas"]["NodeFieldValue"][] | null;
            /**
             * Retried From Item Id
             * @description The item_id of the queue item that this item was retried from
             */
            retried_from_item_id?: number | null;
            /**
             * Is Api Validation Run
             * @description Whether this queue item is an API validation run.
             * @default false
             */
            is_api_validation_run?: boolean;
            /**
             * Published Workflow Id
             * @description The ID of the published workflow associated with this queue item
             */
            published_workflow_id?: string | null;
            /**
             * Api Input Fields
             * @description The fields that were used as input to the API
             */
            api_input_fields?: components["schemas"]["FieldIdentifier"][] | null;
            /**
             * Api Output Fields
             * @description The nodes that were used as output from the API
             */
            api_output_fields?: components["schemas"]["FieldIdentifier"][] | null;
            /** @description The fully-populated session to be executed */
            session: components["schemas"]["GraphExecutionState"];
            /** @description The workflow associated with this queue item */
            workflow?: components["schemas"]["WorkflowWithoutID"] | null;
        };
        /** SessionQueueItemDTO */
        SessionQueueItemDTO: {
            /**
             * Item Id
             * @description The identifier of the session queue item
             */
            item_id: number;
            /**
             * Status
             * @description The status of this queue item
             * @default pending
             * @enum {string}
             */
            status: "pending" | "in_progress" | "completed" | "failed" | "canceled";
            /**
             * Priority
             * @description The priority of this queue item
             * @default 0
             */
            priority: number;
            /**
             * Batch Id
             * @description The ID of the batch associated with this queue item
             */
            batch_id: string;
            /**
             * Origin
             * @description The origin of this queue item. This data is used by the frontend to determine how to handle results.
             */
            origin?: string | null;
            /**
             * Destination
             * @description The origin of this queue item. This data is used by the frontend to determine how to handle results
             */
            destination?: string | null;
            /**
             * Session Id
             * @description The ID of the session associated with this queue item. The session doesn't exist in graph_executions until the queue item is executed.
             */
            session_id: string;
            /**
             * Error Type
             * @description The error type if this queue item errored
             */
            error_type?: string | null;
            /**
             * Error Message
             * @description The error message if this queue item errored
             */
            error_message?: string | null;
            /**
             * Error Traceback
             * @description The error traceback if this queue item errored
             */
            error_traceback?: string | null;
            /**
             * Created At
             * @description When this queue item was created
             */
            created_at: string;
            /**
             * Updated At
             * @description When this queue item was updated
             */
            updated_at: string;
            /**
             * Started At
             * @description When this queue item was started
             */
            started_at?: string | null;
            /**
             * Completed At
             * @description When this queue item was completed
             */
            completed_at?: string | null;
            /**
             * Queue Id
             * @description The id of the queue with which this item is associated
             */
            queue_id: string;
            /**
             * Field Values
             * @description The field values that were used for this queue item
             */
            field_values?: components["schemas"]["NodeFieldValue"][] | null;
            /**
             * Retried From Item Id
             * @description The item_id of the queue item that this item was retried from
             */
            retried_from_item_id?: number | null;
            /**
             * Is Api Validation Run
             * @description Whether this queue item is an API validation run.
             * @default false
             */
            is_api_validation_run?: boolean;
            /**
             * Published Workflow Id
             * @description The ID of the published workflow associated with this queue item
             */
            published_workflow_id?: string | null;
            /**
             * Api Input Fields
             * @description The fields that were used as input to the API
             */
            api_input_fields?: components["schemas"]["FieldIdentifier"][] | null;
            /**
             * Api Output Fields
             * @description The nodes that were used as output from the API
             */
            api_output_fields?: components["schemas"]["FieldIdentifier"][] | null;
        };
        /** SessionQueueStatus */
        SessionQueueStatus: {
            /**
             * Queue Id
             * @description The ID of the queue
             */
            queue_id: string;
            /**
             * Item Id
             * @description The current queue item id
             */
            item_id: number | null;
            /**
             * Batch Id
             * @description The current queue item's batch id
             */
            batch_id: string | null;
            /**
             * Session Id
             * @description The current queue item's session id
             */
            session_id: string | null;
            /**
             * Pending
             * @description Number of queue items with status 'pending'
             */
            pending: number;
            /**
             * In Progress
             * @description Number of queue items with status 'in_progress'
             */
            in_progress: number;
            /**
             * Completed
             * @description Number of queue items with status 'complete'
             */
            completed: number;
            /**
             * Failed
             * @description Number of queue items with status 'error'
             */
            failed: number;
            /**
             * Canceled
             * @description Number of queue items with status 'canceled'
             */
            canceled: number;
            /**
             * Total
             * @description Total number of queue items
             */
            total: number;
        };
        /**
         * Show Image
         * @description Displays a provided image using the OS image viewer, and passes it forward in the pipeline.
         */
        ShowImageInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to show
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * type
             * @default show_image
             * @constant
             */
            type: "show_image";
        };
        /**
         * SigLIPConfig
         * @description Model config for SigLIP.
         */
        SigLIPConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default siglip
             * @constant
             */
            type: "siglip";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
        };
        /**
         * Image-to-Image (Autoscale)
         * @description Run any spandrel image-to-image model (https://github.com/chaiNNer-org/spandrel) until the target scale is reached.
         */
        SpandrelImageToImageAutoscaleInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The input image
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Image-to-Image Model
             * @description Image-to-Image model
             * @default null
             */
            image_to_image_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Tile Size
             * @description The tile size for tiled image-to-image. Set to 0 to disable tiling.
             * @default 512
             */
            tile_size?: number;
            /**
             * type
             * @default spandrel_image_to_image_autoscale
             * @constant
             */
            type: "spandrel_image_to_image_autoscale";
            /**
             * Scale
             * @description The final scale of the output image. If the model does not upscale the image, this will be ignored.
             * @default 4
             */
            scale?: number;
            /**
             * Fit To Multiple Of 8
             * @description If true, the output image will be resized to the nearest multiple of 8 in both dimensions.
             * @default false
             */
            fit_to_multiple_of_8?: boolean;
        };
        /**
         * SpandrelImageToImageConfig
         * @description Model config for Spandrel Image to Image models.
         */
        SpandrelImageToImageConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default spandrel_image_to_image
             * @constant
             */
            type: "spandrel_image_to_image";
            /**
             * Format
             * @default checkpoint
             * @constant
             */
            format: "checkpoint";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /**
         * Image-to-Image
         * @description Run any spandrel image-to-image model (https://github.com/chaiNNer-org/spandrel).
         */
        SpandrelImageToImageInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The input image
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Image-to-Image Model
             * @description Image-to-Image model
             * @default null
             */
            image_to_image_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Tile Size
             * @description The tile size for tiled image-to-image. Set to 0 to disable tiling.
             * @default 512
             */
            tile_size?: number;
            /**
             * type
             * @default spandrel_image_to_image
             * @constant
             */
            type: "spandrel_image_to_image";
        };
        /** StarterModel */
        StarterModel: {
            /** Description */
            description: string;
            /** Source */
            source: string;
            /** Name */
            name: string;
            base: components["schemas"]["BaseModelType"];
            type: components["schemas"]["ModelType"];
            format?: components["schemas"]["ModelFormat"] | null;
            /**
             * Is Installed
             * @default false
             */
            is_installed?: boolean;
            /**
             * Previous Names
             * @default []
             */
            previous_names?: string[];
            /** Dependencies */
            dependencies?: components["schemas"]["StarterModelWithoutDependencies"][] | null;
        };
        /** StarterModelResponse */
        StarterModelResponse: {
            /** Starter Models */
            starter_models: components["schemas"]["StarterModel"][];
            /** Starter Bundles */
            starter_bundles: {
                [key: string]: components["schemas"]["StarterModel"][];
            };
        };
        /** StarterModelWithoutDependencies */
        StarterModelWithoutDependencies: {
            /** Description */
            description: string;
            /** Source */
            source: string;
            /** Name */
            name: string;
            base: components["schemas"]["BaseModelType"];
            type: components["schemas"]["ModelType"];
            format?: components["schemas"]["ModelFormat"] | null;
            /**
             * Is Installed
             * @default false
             */
            is_installed?: boolean;
            /**
             * Previous Names
             * @default []
             */
            previous_names?: string[];
        };
        /**
         * String2Output
         * @description Base class for invocations that output two strings
         */
        String2Output: {
            /**
             * String 1
             * @description string 1
             */
            string_1: string;
            /**
             * String 2
             * @description string 2
             */
            string_2: string;
            /**
             * type
             * @default string_2_output
             * @constant
             */
            type: "string_2_output";
        };
        /**
         * String Batch
         * @description Create a batched generation, where the workflow is executed once for each string in the batch.
         */
        StringBatchInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Batch Group
             * @description The ID of this batch node's group. If provided, all batch nodes in with the same ID will be 'zipped' before execution, and all nodes' collections must be of the same size.
             * @default None
             * @enum {string}
             */
            batch_group_id?: "None" | "Group 1" | "Group 2" | "Group 3" | "Group 4" | "Group 5";
            /**
             * Strings
             * @description The strings to batch over
             * @default []
             */
            strings?: string[];
            /**
             * type
             * @default string_batch
             * @constant
             */
            type: "string_batch";
        };
        /**
         * String Collection Primitive
         * @description A collection of string primitive values
         */
        StringCollectionInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Collection
             * @description The collection of string values
             * @default []
             */
            collection?: string[];
            /**
             * type
             * @default string_collection
             * @constant
             */
            type: "string_collection";
        };
        /**
         * StringCollectionOutput
         * @description Base class for nodes that output a collection of strings
         */
        StringCollectionOutput: {
            /**
             * Collection
             * @description The output strings
             */
            collection: string[];
            /**
             * type
             * @default string_collection_output
             * @constant
             */
            type: "string_collection_output";
        };
        /**
         * String Generator
         * @description Generated a range of strings for use in a batched generation
         */
        StringGenerator: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Generator Type
             * @description The string generator.
             */
            generator: components["schemas"]["StringGeneratorField"];
            /**
             * type
             * @default string_generator
             * @constant
             */
            type: "string_generator";
        };
        /** StringGeneratorField */
        StringGeneratorField: Record<string, never>;
        /**
         * StringGeneratorOutput
         * @description Base class for nodes that output a collection of strings
         */
        StringGeneratorOutput: {
            /**
             * Strings
             * @description The generated strings
             */
            strings: string[];
            /**
             * type
             * @default string_generator_output
             * @constant
             */
            type: "string_generator_output";
        };
        /**
         * String Primitive
         * @description A string primitive value
         */
        StringInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * Value
             * @description The string value
             * @default
             */
            value?: string;
            /**
             * type
             * @default string
             * @constant
             */
            type: "string";
        };
        /**
         * String Join
         * @description Joins string left to string right
         */
        StringJoinInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * String Left
             * @description String Left
             * @default
             */
            string_left?: string;
            /**
             * String Right
             * @description String Right
             * @default
             */
            string_right?: string;
            /**
             * type
             * @default string_join
             * @constant
             */
            type: "string_join";
        };
        /**
         * String Join Three
         * @description Joins string left to string middle to string right
         */
        StringJoinThreeInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * String Left
             * @description String Left
             * @default
             */
            string_left?: string;
            /**
             * String Middle
             * @description String Middle
             * @default
             */
            string_middle?: string;
            /**
             * String Right
             * @description String Right
             * @default
             */
            string_right?: string;
            /**
             * type
             * @default string_join_three
             * @constant
             */
            type: "string_join_three";
        };
        /**
         * StringOutput
         * @description Base class for nodes that output a single string
         */
        StringOutput: {
            /**
             * Value
             * @description The output string
             */
            value: string;
            /**
             * type
             * @default string_output
             * @constant
             */
            type: "string_output";
        };
        /**
         * StringPosNegOutput
         * @description Base class for invocations that output a positive and negative string
         */
        StringPosNegOutput: {
            /**
             * Positive String
             * @description Positive string
             */
            positive_string: string;
            /**
             * Negative String
             * @description Negative string
             */
            negative_string: string;
            /**
             * type
             * @default string_pos_neg_output
             * @constant
             */
            type: "string_pos_neg_output";
        };
        /**
         * String Replace
         * @description Replaces the search string with the replace string
         */
        StringReplaceInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * String
             * @description String to work on
             * @default
             */
            string?: string;
            /**
             * Search String
             * @description String to search for
             * @default
             */
            search_string?: string;
            /**
             * Replace String
             * @description String to replace the search
             * @default
             */
            replace_string?: string;
            /**
             * Use Regex
             * @description Use search string as a regex expression (non regex is case insensitive)
             * @default false
             */
            use_regex?: boolean;
            /**
             * type
             * @default string_replace
             * @constant
             */
            type: "string_replace";
        };
        /**
         * String Split
         * @description Splits string into two strings, based on the first occurance of the delimiter. The delimiter will be removed from the string
         */
        StringSplitInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * String
             * @description String to split
             * @default
             */
            string?: string;
            /**
             * Delimiter
             * @description Delimiter to spilt with. blank will split on the first whitespace
             * @default
             */
            delimiter?: string;
            /**
             * type
             * @default string_split
             * @constant
             */
            type: "string_split";
        };
        /**
         * String Split Negative
         * @description Splits string into two strings, inside [] goes into negative string everthing else goes into positive string. Each [ and ] character is replaced with a space
         */
        StringSplitNegInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * String
             * @description String to split
             * @default
             */
            string?: string;
            /**
             * type
             * @default string_split_neg
             * @constant
             */
            type: "string_split_neg";
        };
        /** StylePresetRecordWithImage */
        StylePresetRecordWithImage: {
            /**
             * Name
             * @description The name of the style preset.
             */
            name: string;
            /** @description The preset data */
            preset_data: components["schemas"]["PresetData"];
            /** @description The type of style preset */
            type: components["schemas"]["PresetType"];
            /**
             * Id
             * @description The style preset ID.
             */
            id: string;
            /**
             * Image
             * @description The path for image
             */
            image: string | null;
        };
        /**
         * SubModelType
         * @description Submodel type.
         * @enum {string}
         */
        SubModelType: "unet" | "transformer" | "text_encoder" | "text_encoder_2" | "text_encoder_3" | "tokenizer" | "tokenizer_2" | "tokenizer_3" | "vae" | "vae_decoder" | "vae_encoder" | "scheduler" | "safety_checker";
        /** SubmodelDefinition */
        SubmodelDefinition: {
            /** Path Or Prefix */
            path_or_prefix: string;
            model_type: components["schemas"]["ModelType"];
            /** Variant */
            variant?: components["schemas"]["ModelVariantType"] | components["schemas"]["ClipVariantType"] | null;
        };
        /**
         * Subtract Integers
         * @description Subtracts two numbers
         */
        SubtractInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * A
             * @description The first number
             * @default 0
             */
            a?: number;
            /**
             * B
             * @description The second number
             * @default 0
             */
            b?: number;
            /**
             * type
             * @default sub
             * @constant
             */
            type: "sub";
        };
        /**
         * T2IAdapterConfig
         * @description Model config for T2I.
         */
        T2IAdapterConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default t2i_adapter
             * @constant
             */
            type: "t2i_adapter";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /** @description Default settings for this model */
            default_settings?: components["schemas"]["ControlAdapterDefaultSettings"] | null;
            /** @default  */
            repo_variant?: components["schemas"]["ModelRepoVariant"] | null;
        };
        /** T2IAdapterField */
        T2IAdapterField: {
            /** @description The T2I-Adapter image prompt. */
            image: components["schemas"]["ImageField"];
            /** @description The T2I-Adapter model to use. */
            t2i_adapter_model: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight given to the T2I-Adapter
             * @default 1
             */
            weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the T2I-Adapter is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the T2I-Adapter is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Resize Mode
             * @description The resize mode to use
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
        };
        /**
         * T2I-Adapter - SD1.5, SDXL
         * @description Collects T2I-Adapter info to pass to other nodes.
         */
        T2IAdapterInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The IP-Adapter image prompt.
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * T2I-Adapter Model
             * @description The T2I-Adapter model.
             * @default null
             */
            t2i_adapter_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight given to the T2I-Adapter
             * @default 1
             */
            weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the T2I-Adapter is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the T2I-Adapter is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Resize Mode
             * @description The resize mode applied to the T2I-Adapter input image so that it matches the target output size.
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
            /**
             * type
             * @default t2i_adapter
             * @constant
             */
            type: "t2i_adapter";
        };
        /** T2IAdapterMetadataField */
        T2IAdapterMetadataField: {
            /** @description The control image. */
            image: components["schemas"]["ImageField"];
            /**
             * @description The control image, after processing.
             * @default null
             */
            processed_image?: components["schemas"]["ImageField"] | null;
            /** @description The T2I-Adapter model to use. */
            t2i_adapter_model: components["schemas"]["ModelIdentifierField"];
            /**
             * Weight
             * @description The weight given to the T2I-Adapter
             * @default 1
             */
            weight?: number | number[];
            /**
             * Begin Step Percent
             * @description When the T2I-Adapter is first applied (% of total steps)
             * @default 0
             */
            begin_step_percent?: number;
            /**
             * End Step Percent
             * @description When the T2I-Adapter is last applied (% of total steps)
             * @default 1
             */
            end_step_percent?: number;
            /**
             * Resize Mode
             * @description The resize mode to use
             * @default just_resize
             * @enum {string}
             */
            resize_mode?: "just_resize" | "crop_resize" | "fill_resize" | "just_resize_simple";
        };
        /** T2IAdapterOutput */
        T2IAdapterOutput: {
            /**
             * T2I Adapter
             * @description T2I-Adapter(s) to apply
             */
            t2i_adapter: components["schemas"]["T2IAdapterField"];
            /**
             * type
             * @default t2i_adapter_output
             * @constant
             */
            type: "t2i_adapter_output";
        };
        /** T5EncoderBnbQuantizedLlmInt8bConfig */
        T5EncoderBnbQuantizedLlmInt8bConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default t5_encoder
             * @constant
             */
            type: "t5_encoder";
            /**
             * Format
             * @default bnb_quantized_int8b
             * @constant
             */
            format: "bnb_quantized_int8b";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /** T5EncoderConfig */
        T5EncoderConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default t5_encoder
             * @constant
             */
            type: "t5_encoder";
            /**
             * Format
             * @default t5_encoder
             * @constant
             */
            format: "t5_encoder";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /** T5EncoderField */
        T5EncoderField: {
            /** @description Info to load tokenizer submodel */
            tokenizer: components["schemas"]["ModelIdentifierField"];
            /** @description Info to load text_encoder submodel */
            text_encoder: components["schemas"]["ModelIdentifierField"];
            /**
             * Loras
             * @description LoRAs to apply on model loading
             */
            loras: components["schemas"]["LoRAField"][];
        };
        /** TBLR */
        TBLR: {
            /** Top */
            top: number;
            /** Bottom */
            bottom: number;
            /** Left */
            left: number;
            /** Right */
            right: number;
        };
        /**
         * TensorField
         * @description A tensor primitive field.
         */
        TensorField: {
            /**
             * Tensor Name
             * @description The name of a tensor.
             */
            tensor_name: string;
        };
        /**
         * TextualInversionFileConfig
         * @description Model config for textual inversion embeddings.
         */
        TextualInversionFileConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default embedding
             * @constant
             */
            type: "embedding";
            /**
             * Format
             * @default embedding_file
             * @constant
             */
            format: "embedding_file";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /**
         * TextualInversionFolderConfig
         * @description Model config for textual inversion embeddings.
         */
        TextualInversionFolderConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default embedding
             * @constant
             */
            type: "embedding";
            /**
             * Format
             * @default embedding_folder
             * @constant
             */
            format: "embedding_folder";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /** Tile */
        Tile: {
            /** @description The coordinates of this tile relative to its parent image. */
            coords: components["schemas"]["TBLR"];
            /** @description The amount of overlap with adjacent tiles on each side of this tile. */
            overlap: components["schemas"]["TBLR"];
        };
        /**
         * Tile to Properties
         * @description Split a Tile into its individual properties.
         */
        TileToPropertiesInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The tile to split into properties.
             * @default null
             */
            tile?: components["schemas"]["Tile"];
            /**
             * type
             * @default tile_to_properties
             * @constant
             */
            type: "tile_to_properties";
        };
        /** TileToPropertiesOutput */
        TileToPropertiesOutput: {
            /**
             * Coords Left
             * @description Left coordinate of the tile relative to its parent image.
             */
            coords_left: number;
            /**
             * Coords Right
             * @description Right coordinate of the tile relative to its parent image.
             */
            coords_right: number;
            /**
             * Coords Top
             * @description Top coordinate of the tile relative to its parent image.
             */
            coords_top: number;
            /**
             * Coords Bottom
             * @description Bottom coordinate of the tile relative to its parent image.
             */
            coords_bottom: number;
            /**
             * Width
             * @description The width of the tile. Equal to coords_right - coords_left.
             */
            width: number;
            /**
             * Height
             * @description The height of the tile. Equal to coords_bottom - coords_top.
             */
            height: number;
            /**
             * Overlap Top
             * @description Overlap between this tile and its top neighbor.
             */
            overlap_top: number;
            /**
             * Overlap Bottom
             * @description Overlap between this tile and its bottom neighbor.
             */
            overlap_bottom: number;
            /**
             * Overlap Left
             * @description Overlap between this tile and its left neighbor.
             */
            overlap_left: number;
            /**
             * Overlap Right
             * @description Overlap between this tile and its right neighbor.
             */
            overlap_right: number;
            /**
             * type
             * @default tile_to_properties_output
             * @constant
             */
            type: "tile_to_properties_output";
        };
        /** TileWithImage */
        TileWithImage: {
            tile: components["schemas"]["Tile"];
            image: components["schemas"]["ImageField"];
        };
        /**
         * Tiled Multi-Diffusion Denoise - SD1.5, SDXL
         * @description Tiled Multi-Diffusion denoising.
         *
         *     This node handles automatically tiling the input image, and is primarily intended for global refinement of images
         *     in tiled upscaling workflows. Future Multi-Diffusion nodes should allow the user to specify custom regions with
         *     different parameters for each region to harness the full power of Multi-Diffusion.
         *
         *     This node has a similar interface to the `DenoiseLatents` node, but it has a reduced feature set (no IP-Adapter,
         *     T2I-Adapter, masking, etc.).
         */
        TiledMultiDiffusionDenoiseLatents: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description Positive conditioning tensor
             * @default null
             */
            positive_conditioning?: components["schemas"]["ConditioningField"];
            /**
             * @description Negative conditioning tensor
             * @default null
             */
            negative_conditioning?: components["schemas"]["ConditioningField"];
            /**
             * @description Noise tensor
             * @default null
             */
            noise?: components["schemas"]["LatentsField"] | null;
            /**
             * @description Latents tensor
             * @default null
             */
            latents?: components["schemas"]["LatentsField"] | null;
            /**
             * Tile Height
             * @description Height of the tiles in image space.
             * @default 1024
             */
            tile_height?: number;
            /**
             * Tile Width
             * @description Width of the tiles in image space.
             * @default 1024
             */
            tile_width?: number;
            /**
             * Tile Overlap
             * @description The overlap between adjacent tiles in pixel space. (Of course, tile merging is applied in latent space.) Tiles will be cropped during merging (if necessary) to ensure that they overlap by exactly this amount.
             * @default 32
             */
            tile_overlap?: number;
            /**
             * Steps
             * @description Number of steps to run
             * @default 18
             */
            steps?: number;
            /**
             * CFG Scale
             * @description Classifier-Free Guidance scale
             * @default 6
             */
            cfg_scale?: number | number[];
            /**
             * Denoising Start
             * @description When to start denoising, expressed a percentage of total steps
             * @default 0
             */
            denoising_start?: number;
            /**
             * Denoising End
             * @description When to stop denoising, expressed a percentage of total steps
             * @default 1
             */
            denoising_end?: number;
            /**
             * Scheduler
             * @description Scheduler to use during inference
             * @default euler
             * @enum {string}
             */
            scheduler?: "ddim" | "ddpm" | "deis" | "deis_k" | "lms" | "lms_k" | "pndm" | "heun" | "heun_k" | "euler" | "euler_k" | "euler_a" | "kdpm_2" | "kdpm_2_k" | "kdpm_2_a" | "kdpm_2_a_k" | "dpmpp_2s" | "dpmpp_2s_k" | "dpmpp_2m" | "dpmpp_2m_k" | "dpmpp_2m_sde" | "dpmpp_2m_sde_k" | "dpmpp_3m" | "dpmpp_3m_k" | "dpmpp_sde" | "dpmpp_sde_k" | "unipc" | "unipc_k" | "lcm" | "tcd";
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             * @default null
             */
            unet?: components["schemas"]["UNetField"];
            /**
             * CFG Rescale Multiplier
             * @description Rescale multiplier for CFG guidance, used for models trained with zero-terminal SNR
             * @default 0
             */
            cfg_rescale_multiplier?: number;
            /**
             * Control
             * @default null
             */
            control?: components["schemas"]["ControlField"] | components["schemas"]["ControlField"][] | null;
            /**
             * type
             * @default tiled_multi_diffusion_denoise_latents
             * @constant
             */
            type: "tiled_multi_diffusion_denoise_latents";
        };
        /** TransformerField */
        TransformerField: {
            /** @description Info to load Transformer submodel */
            transformer: components["schemas"]["ModelIdentifierField"];
            /**
             * Loras
             * @description LoRAs to apply on model loading
             */
            loras: components["schemas"]["LoRAField"][];
        };
        /**
         * UIComponent
         * @description The type of UI component to use for a field, used to override the default components, which are
         *     inferred from the field type.
         * @enum {string}
         */
        UIComponent: "none" | "textarea" | "slider";
        /**
         * UIConfigBase
         * @description Provides additional node configuration to the UI.
         *     This is used internally by the @invocation decorator logic. Do not use this directly.
         */
        UIConfigBase: {
            /**
             * Tags
             * @description The node's tags
             */
            tags: string[] | null;
            /**
             * Title
             * @description The node's display name
             * @default null
             */
            title: string | null;
            /**
             * Category
             * @description The node's category
             * @default null
             */
            category: string | null;
            /**
             * Version
             * @description The node's version. Should be a valid semver string e.g. "1.0.0" or "3.8.13".
             */
            version: string;
            /**
             * Node Pack
             * @description The node pack that this node belongs to, will be 'invokeai' for built-in nodes
             */
            node_pack: string;
            /**
             * @description The node's classification
             * @default stable
             */
            classification: components["schemas"]["Classification"];
        };
        /**
         * UIType
         * @description Type hints for the UI for situations in which the field type is not enough to infer the correct UI type.
         *
         *     - Model Fields
         *     The most common node-author-facing use will be for model fields. Internally, there is no difference
         *     between SD-1, SD-2 and SDXL model fields - they all use the class `MainModelField`. To ensure the
         *     base-model-specific UI is rendered, use e.g. `ui_type=UIType.SDXLMainModelField` to indicate that
         *     the field is an SDXL main model field.
         *
         *     - Any Field
         *     We cannot infer the usage of `typing.Any` via schema parsing, so you *must* use `ui_type=UIType.Any` to
         *     indicate that the field accepts any type. Use with caution. This cannot be used on outputs.
         *
         *     - Scheduler Field
         *     Special handling in the UI is needed for this field, which otherwise would be parsed as a plain enum field.
         *
         *     - Internal Fields
         *     Similar to the Any Field, the `collect` and `iterate` nodes make use of `typing.Any`. To facilitate
         *     handling these types in the client, we use `UIType._Collection` and `UIType._CollectionItem`. These
         *     should not be used by node authors.
         *
         *     - DEPRECATED Fields
         *     These types are deprecated and should not be used by node authors. A warning will be logged if one is
         *     used, and the type will be ignored. They are included here for backwards compatibility.
         * @enum {string}
         */
        UIType: "MainModelField" | "CogView4MainModelField" | "FluxMainModelField" | "SD3MainModelField" | "SDXLMainModelField" | "SDXLRefinerModelField" | "ONNXModelField" | "VAEModelField" | "FluxVAEModelField" | "LoRAModelField" | "ControlNetModelField" | "IPAdapterModelField" | "T2IAdapterModelField" | "T5EncoderModelField" | "CLIPEmbedModelField" | "CLIPLEmbedModelField" | "CLIPGEmbedModelField" | "SpandrelImageToImageModelField" | "ControlLoRAModelField" | "SigLipModelField" | "FluxReduxModelField" | "LLaVAModelField" | "SchedulerField" | "AnyField" | "CollectionField" | "CollectionItemField" | "DEPRECATED_Boolean" | "DEPRECATED_Color" | "DEPRECATED_Conditioning" | "DEPRECATED_Control" | "DEPRECATED_Float" | "DEPRECATED_Image" | "DEPRECATED_Integer" | "DEPRECATED_Latents" | "DEPRECATED_String" | "DEPRECATED_BooleanCollection" | "DEPRECATED_ColorCollection" | "DEPRECATED_ConditioningCollection" | "DEPRECATED_ControlCollection" | "DEPRECATED_FloatCollection" | "DEPRECATED_ImageCollection" | "DEPRECATED_IntegerCollection" | "DEPRECATED_LatentsCollection" | "DEPRECATED_StringCollection" | "DEPRECATED_BooleanPolymorphic" | "DEPRECATED_ColorPolymorphic" | "DEPRECATED_ConditioningPolymorphic" | "DEPRECATED_ControlPolymorphic" | "DEPRECATED_FloatPolymorphic" | "DEPRECATED_ImagePolymorphic" | "DEPRECATED_IntegerPolymorphic" | "DEPRECATED_LatentsPolymorphic" | "DEPRECATED_StringPolymorphic" | "DEPRECATED_UNet" | "DEPRECATED_Vae" | "DEPRECATED_CLIP" | "DEPRECATED_Collection" | "DEPRECATED_CollectionItem" | "DEPRECATED_Enum" | "DEPRECATED_WorkflowField" | "DEPRECATED_IsIntermediate" | "DEPRECATED_BoardField" | "DEPRECATED_MetadataItem" | "DEPRECATED_MetadataItemCollection" | "DEPRECATED_MetadataItemPolymorphic" | "DEPRECATED_MetadataDict";
        /** UNetField */
        UNetField: {
            /** @description Info to load unet submodel */
            unet: components["schemas"]["ModelIdentifierField"];
            /** @description Info to load scheduler submodel */
            scheduler: components["schemas"]["ModelIdentifierField"];
            /**
             * Loras
             * @description LoRAs to apply on model loading
             */
            loras: components["schemas"]["LoRAField"][];
            /**
             * Seamless Axes
             * @description Axes("x" and "y") to which apply seamless
             */
            seamless_axes?: string[];
            /**
             * @description FreeU configuration
             * @default null
             */
            freeu_config?: components["schemas"]["FreeUConfig"] | null;
        };
        /**
         * UNetOutput
         * @description Base class for invocations that output a UNet field.
         */
        UNetOutput: {
            /**
             * UNet
             * @description UNet (scheduler, LoRAs)
             */
            unet: components["schemas"]["UNetField"];
            /**
             * type
             * @default unet_output
             * @constant
             */
            type: "unet_output";
        };
        /**
         * URLModelSource
         * @description A generic URL point to a checkpoint file.
         */
        URLModelSource: {
            /**
             * Url
             * Format: uri
             */
            url: string;
            /** Access Token */
            access_token?: string | null;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            type: "url";
        };
        /** URLRegexTokenPair */
        URLRegexTokenPair: {
            /**
             * Url Regex
             * @description Regular expression to match against the URL
             */
            url_regex: string;
            /**
             * Token
             * @description Token to use when the URL matches the regex
             */
            token: string;
        };
        /**
         * Unsharp Mask
         * @description Applies an unsharp mask filter to an image
         */
        UnsharpMaskInvocation: {
            /**
             * @description The board to save the image to
             * @default null
             */
            board?: components["schemas"]["BoardField"] | null;
            /**
             * @description Optional metadata to be saved with the image
             * @default null
             */
            metadata?: components["schemas"]["MetadataField"] | null;
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * @description The image to use
             * @default null
             */
            image?: components["schemas"]["ImageField"];
            /**
             * Radius
             * @description Unsharp mask radius
             * @default 2
             */
            radius?: number;
            /**
             * Strength
             * @description Unsharp mask strength
             * @default 50
             */
            strength?: number;
            /**
             * type
             * @default unsharp_mask
             * @constant
             */
            type: "unsharp_mask";
        };
        /** Upscaler */
        Upscaler: {
            /**
             * Upscaling Method
             * @description Name of upscaling method
             */
            upscaling_method: string;
            /**
             * Upscaling Models
             * @description List of upscaling models for this method
             */
            upscaling_models: string[];
        };
        /**
         * VAECheckpointConfig
         * @description Model config for standalone VAE models.
         */
        VAECheckpointConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default vae
             * @constant
             */
            type: "vae";
            /**
             * Format
             * @description Format of the provided checkpoint model
             * @default checkpoint
             * @enum {string}
             */
            format: "checkpoint" | "bnb_quantized_nf4b" | "gguf_quantized";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
            /**
             * Config Path
             * @description path to the checkpoint model config file
             */
            config_path: string;
            /**
             * Converted At
             * @description When this model was last converted to diffusers
             */
            converted_at?: number | null;
        };
        /**
         * VAEDiffusersConfig
         * @description Model config for standalone VAE models (diffusers version).
         */
        VAEDiffusersConfig: {
            /**
             * Key
             * @description A unique key for this model.
             */
            key: string;
            /**
             * Hash
             * @description The hash of the model file(s).
             */
            hash: string;
            /**
             * Path
             * @description Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.
             */
            path: string;
            /**
             * File Size
             * @description The size of the model in bytes.
             */
            file_size: number;
            /**
             * Name
             * @description Name of the model.
             */
            name: string;
            /**
             * Type
             * @default vae
             * @constant
             */
            type: "vae";
            /**
             * Format
             * @default diffusers
             * @constant
             */
            format: "diffusers";
            /** @description The base model. */
            base: components["schemas"]["BaseModelType"];
            /**
             * Source
             * @description The original source of the model (path, URL or repo_id).
             */
            source: string;
            /** @description The type of source */
            source_type: components["schemas"]["ModelSourceType"];
            /**
             * Description
             * @description Model description
             */
            description?: string | null;
            /**
             * Source Api Response
             * @description The original API response from the source, as stringified JSON.
             */
            source_api_response?: string | null;
            /**
             * Cover Image
             * @description Url for image to preview model
             */
            cover_image?: string | null;
            /**
             * Submodels
             * @description Loadable submodels in this model
             */
            submodels?: {
                [key: string]: components["schemas"]["SubmodelDefinition"];
            } | null;
        };
        /** VAEField */
        VAEField: {
            /** @description Info to load vae submodel */
            vae: components["schemas"]["ModelIdentifierField"];
            /**
             * Seamless Axes
             * @description Axes("x" and "y") to which apply seamless
             */
            seamless_axes?: string[];
        };
        /**
         * VAE Model - SD1.5, SDXL, SD3, FLUX
         * @description Loads a VAE model, outputting a VaeLoaderOutput
         */
        VAELoaderInvocation: {
            /**
             * Id
             * @description The id of this instance of an invocation. Must be unique among all instances of invocations.
             */
            id: string;
            /**
             * Is Intermediate
             * @description Whether or not this is an intermediate invocation.
             * @default false
             */
            is_intermediate?: boolean;
            /**
             * Use Cache
             * @description Whether or not to use the cache
             * @default true
             */
            use_cache?: boolean;
            /**
             * VAE
             * @description VAE model to load
             * @default null
             */
            vae_model?: components["schemas"]["ModelIdentifierField"];
            /**
             * type
             * @default vae_loader
             * @constant
             */
            type: "vae_loader";
        };
        /**
         * VAEOutput
         * @description Base class for invocations that output a VAE field
         */
        VAEOutput: {
            /**
             * VAE
             * @description VAE
             */
            vae: components["schemas"]["VAEField"];
            /**
             * type
             * @default vae_output
             * @constant
             */
            type: "vae_output";
        };
        /** ValidationError */
        ValidationError: {
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
        /** ValidationRunData */
        ValidationRunData: {
            /**
             * Workflow Id
             * @description The id of the workflow being published.
             */
            workflow_id: string;
            /**
             * Input Fields
             * @description The input fields for the published workflow
             */
            input_fields: components["schemas"]["FieldIdentifier"][];
            /**
             * Output Fields
             * @description The output fields for the published workflow
             */
            output_fields: components["schemas"]["FieldIdentifier"][];
        };
        /** Workflow */
        Workflow: {
            /**
             * Name
             * @description The name of the workflow.
             */
            name: string;
            /**
             * Author
             * @description The author of the workflow.
             */
            author: string;
            /**
             * Description
             * @description The description of the workflow.
             */
            description: string;
            /**
             * Version
             * @description The version of the workflow.
             */
            version: string;
            /**
             * Contact
             * @description The contact of the workflow.
             */
            contact: string;
            /**
             * Tags
             * @description The tags of the workflow.
             */
            tags: string;
            /**
             * Notes
             * @description The notes of the workflow.
             */
            notes: string;
            /**
             * Exposedfields
             * @description The exposed fields of the workflow.
             */
            exposedFields: components["schemas"]["ExposedField"][];
            /** @description The meta of the workflow. */
            meta: components["schemas"]["WorkflowMeta"];
            /**
             * Nodes
             * @description The nodes of the workflow.
             */
            nodes: {
                [key: string]: components["schemas"]["JsonValue"];
            }[];
            /**
             * Edges
             * @description The edges of the workflow.
             */
            edges: {
                [key: string]: components["schemas"]["JsonValue"];
            }[];
            /**
             * Form
             * @description The form of the workflow.
             */
            form?: {
                [key: string]: components["schemas"]["JsonValue"];
            } | null;
            /**
             * Is Published
             * @description Whether the workflow is published or not.
             */
            is_published?: boolean | null;
            /**
             * Id
             * @description The id of the workflow.
             */
            id: string;
        };
        /** WorkflowAndGraphResponse */
        WorkflowAndGraphResponse: {
            /**
             * Workflow
             * @description The workflow used to generate the image, as stringified JSON
             */
            workflow: string | null;
            /**
             * Graph
             * @description The graph used to generate the image, as stringified JSON
             */
            graph: string | null;
        };
        /**
         * WorkflowCategory
         * @enum {string}
         */
        WorkflowCategory: "user" | "default" | "project";
        /** WorkflowMeta */
        WorkflowMeta: {
            /**
             * Version
             * @description The version of the workflow schema.
             */
            version: string;
            /** @description The category of the workflow (user or default). */
            category: components["schemas"]["WorkflowCategory"];
        };
        /** WorkflowRecordDTO */
        WorkflowRecordDTO: {
            /**
             * Workflow Id
             * @description The id of the workflow.
             */
            workflow_id: string;
            /**
             * Name
             * @description The name of the workflow.
             */
            name: string;
            /**
             * Created At
             * @description The created timestamp of the workflow.
             */
            created_at: string;
            /**
             * Updated At
             * @description The updated timestamp of the workflow.
             */
            updated_at: string;
            /**
             * Opened At
             * @description The opened timestamp of the workflow.
             */
            opened_at?: string | null;
            /**
             * Is Published
             * @description Whether the workflow is published or not.
             */
            is_published?: boolean | null;
            /** @description The workflow. */
            workflow: components["schemas"]["Workflow"];
        };
        /** WorkflowRecordListItemWithThumbnailDTO */
        WorkflowRecordListItemWithThumbnailDTO: {
            /**
             * Workflow Id
             * @description The id of the workflow.
             */
            workflow_id: string;
            /**
             * Name
             * @description The name of the workflow.
             */
            name: string;
            /**
             * Created At
             * @description The created timestamp of the workflow.
             */
            created_at: string;
            /**
             * Updated At
             * @description The updated timestamp of the workflow.
             */
            updated_at: string;
            /**
             * Opened At
             * @description The opened timestamp of the workflow.
             */
            opened_at?: string | null;
            /**
             * Is Published
             * @description Whether the workflow is published or not.
             */
            is_published?: boolean | null;
            /**
             * Description
             * @description The description of the workflow.
             */
            description: string;
            /** @description The description of the workflow. */
            category: components["schemas"]["WorkflowCategory"];
            /**
             * Tags
             * @description The tags of the workflow.
             */
            tags: string;
            /**
             * Thumbnail Url
             * @description The URL of the workflow thumbnail.
             */
            thumbnail_url?: string | null;
        };
        /**
         * WorkflowRecordOrderBy
         * @description The order by options for workflow records
         * @enum {string}
         */
        WorkflowRecordOrderBy: "created_at" | "updated_at" | "opened_at" | "name";
        /** WorkflowRecordWithThumbnailDTO */
        WorkflowRecordWithThumbnailDTO: {
            /**
             * Workflow Id
             * @description The id of the workflow.
             */
            workflow_id: string;
            /**
             * Name
             * @description The name of the workflow.
             */
            name: string;
            /**
             * Created At
             * @description The created timestamp of the workflow.
             */
            created_at: string;
            /**
             * Updated At
             * @description The updated timestamp of the workflow.
             */
            updated_at: string;
            /**
             * Opened At
             * @description The opened timestamp of the workflow.
             */
            opened_at?: string | null;
            /**
             * Is Published
             * @description Whether the workflow is published or not.
             */
            is_published?: boolean | null;
            /** @description The workflow. */
            workflow: components["schemas"]["Workflow"];
            /**
             * Thumbnail Url
             * @description The URL of the workflow thumbnail.
             */
            thumbnail_url?: string | null;
        };
        /** WorkflowWithoutID */
        WorkflowWithoutID: {
            /**
             * Name
             * @description The name of the workflow.
             */
            name: string;
            /**
             * Author
             * @description The author of the workflow.
             */
            author: string;
            /**
             * Description
             * @description The description of the workflow.
             */
            description: string;
            /**
             * Version
             * @description The version of the workflow.
             */
            version: string;
            /**
             * Contact
             * @description The contact of the workflow.
             */
            contact: string;
            /**
             * Tags
             * @description The tags of the workflow.
             */
            tags: string;
            /**
             * Notes
             * @description The notes of the workflow.
             */
            notes: string;
            /**
             * Exposedfields
             * @description The exposed fields of the workflow.
             */
            exposedFields: components["schemas"]["ExposedField"][];
            /** @description The meta of the workflow. */
            meta: components["schemas"]["WorkflowMeta"];
            /**
             * Nodes
             * @description The nodes of the workflow.
             */
            nodes: {
                [key: string]: components["schemas"]["JsonValue"];
            }[];
            /**
             * Edges
             * @description The edges of the workflow.
             */
            edges: {
                [key: string]: components["schemas"]["JsonValue"];
            }[];
            /**
             * Form
             * @description The form of the workflow.
             */
            form?: {
                [key: string]: components["schemas"]["JsonValue"];
            } | null;
            /**
             * Is Published
             * @description Whether the workflow is published or not.
             */
            is_published?: boolean | null;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
};
export type $defs = Record<string, never>;
export interface operations {
    parse_dynamicprompts: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_parse_dynamicprompts"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DynamicPromptsResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_model_records: {
        parameters: {
            query?: {
                /** @description Base models to include */
                base_models?: components["schemas"]["BaseModelType"][] | null;
                /** @description The type of model to get */
                model_type?: components["schemas"]["ModelType"] | null;
                /** @description Exact match on the name of the model */
                model_name?: string | null;
                /** @description Exact match on the format of the model (e.g. 'diffusers') */
                model_format?: components["schemas"]["ModelFormat"] | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelsList"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_records_by_attrs: {
        parameters: {
            query: {
                /** @description The name of the model */
                name: string;
                /** @description The type of the model */
                type: components["schemas"]["ModelType"];
                /** @description The base model of the model */
                base: components["schemas"]["BaseModelType"];
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_record: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Key of the model record to fetch. */
                key: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The model configuration was retrieved successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"];
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description The model could not be found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_model: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Unique key of model to remove from model registry. */
                key: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Model deleted successfully */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Model not found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_model_record: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Unique key of model */
                key: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                /** @example {
                 *       "path": "/path/to/model",
                 *       "name": "model_name",
                 *       "base": "sd-1",
                 *       "type": "main",
                 *       "format": "checkpoint",
                 *       "config_path": "configs/stable-diffusion/v1-inference.yaml",
                 *       "description": "Model description",
                 *       "variant": "normal"
                 *     } */
                "application/json": components["schemas"]["ModelRecordChanges"];
            };
        };
        responses: {
            /** @description The model was updated successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"];
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description The model could not be found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description There is already a model corresponding to the new name */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    scan_for_models: {
        parameters: {
            query?: {
                /** @description Directory path to search for models */
                scan_path?: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Directory scanned successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["FoundModel"][];
                };
            };
            /** @description Invalid directory path */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_hugging_face_models: {
        parameters: {
            query?: {
                /** @description Hugging face repo to search for models */
                hugging_face_repo?: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Hugging Face repo scanned successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HuggingFaceModels"];
                };
            };
            /** @description Invalid hugging face repo */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_image: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of model image file to get */
                key: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The model image was fetched successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description The model image could not be found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_model_image: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Unique key of model image to remove from model_images directory. */
                key: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Model image deleted successfully */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Model image not found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_model_image: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Unique key of model */
                key: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_update_model_image"];
            };
        };
        responses: {
            /** @description The model image was updated successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_model_installs: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelInstallJob"][];
                };
            };
        };
    };
    install_model: {
        parameters: {
            query: {
                /** @description Model source to install, can be a local path, repo_id, or remote URL */
                source: string;
                /** @description Whether or not to install a local model in place */
                inplace?: boolean | null;
                /** @description access token for the remote resource */
                access_token?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                /** @example {
                 *       "name": "string",
                 *       "description": "string"
                 *     } */
                "application/json": components["schemas"]["ModelRecordChanges"];
            };
        };
        responses: {
            /** @description The model imported successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelInstallJob"];
                };
            };
            /** @description There is already a model corresponding to this path or repo_id */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Unrecognized file/folder format */
            415: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
            /** @description The model appeared to import successfully, but could not be found in the model manager */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    prune_model_install_jobs: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description All completed and errored jobs have been pruned */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    install_hugging_face_model: {
        parameters: {
            query: {
                /** @description HuggingFace repo_id to install */
                source: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The model is being installed */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "text/html": string;
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description There is already a model corresponding to this path or repo_id */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_install_job: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Model install id */
                id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Success */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelInstallJob"];
                };
            };
            /** @description No such job */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_model_install_job: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Model install job ID */
                id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The job was cancelled successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description No such job */
            415: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    convert_model: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description Unique key of the safetensors main model to convert to diffusers format. */
                key: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Model converted successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MainDiffusersConfig"] | components["schemas"]["MainCheckpointConfig"] | components["schemas"]["MainBnbQuantized4bCheckpointConfig"] | components["schemas"]["MainGGUFCheckpointConfig"] | components["schemas"]["VAEDiffusersConfig"] | components["schemas"]["VAECheckpointConfig"] | components["schemas"]["ControlNetDiffusersConfig"] | components["schemas"]["ControlNetCheckpointConfig"] | components["schemas"]["LoRALyCORISConfig"] | components["schemas"]["ControlLoRALyCORISConfig"] | components["schemas"]["ControlLoRADiffusersConfig"] | components["schemas"]["LoRADiffusersConfig"] | components["schemas"]["T5EncoderConfig"] | components["schemas"]["T5EncoderBnbQuantizedLlmInt8bConfig"] | components["schemas"]["TextualInversionFileConfig"] | components["schemas"]["TextualInversionFolderConfig"] | components["schemas"]["IPAdapterInvokeAIConfig"] | components["schemas"]["IPAdapterCheckpointConfig"] | components["schemas"]["T2IAdapterConfig"] | components["schemas"]["SpandrelImageToImageConfig"] | components["schemas"]["CLIPVisionDiffusersConfig"] | components["schemas"]["CLIPLEmbedDiffusersConfig"] | components["schemas"]["CLIPGEmbedDiffusersConfig"] | components["schemas"]["SigLIPConfig"] | components["schemas"]["FluxReduxConfig"] | components["schemas"]["LlavaOnevisionConfig"];
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Model not found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description There is already a model registered at this location */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_starter_models: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StarterModelResponse"];
                };
            };
        };
    };
    get_stats: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CacheStats"] | null;
                };
            };
        };
    };
    empty_model_cache: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    get_hf_login_status: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HFTokenStatus"];
                };
            };
        };
    };
    do_hf_login: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_do_hf_login"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HFTokenStatus"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_downloads: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DownloadJob"][];
                };
            };
        };
    };
    prune_downloads: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description All completed jobs have been pruned */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    download: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_download"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DownloadJob"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_download_job: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description ID of the download job to fetch. */
                id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Success */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DownloadJob"];
                };
            };
            /** @description The requested download JobID could not be found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_download_job: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description ID of the download job to cancel. */
                id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Job has been cancelled */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description The requested download JobID could not be found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_all_download_jobs: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Download jobs have been cancelled */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    upload_image: {
        parameters: {
            query: {
                /** @description The category of the image */
                image_category: components["schemas"]["ImageCategory"];
                /** @description Whether this is an intermediate image */
                is_intermediate: boolean;
                /** @description The board to add this image to, if any */
                board_id?: string | null;
                /** @description The session ID associated with this upload, if any */
                session_id?: string | null;
                /** @description Whether to crop the image */
                crop_visible?: boolean | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_upload_image"];
            };
        };
        responses: {
            /** @description The image was uploaded successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImageDTO"];
                };
            };
            /** @description Image upload failed */
            415: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_image_dtos: {
        parameters: {
            query?: {
                /** @description The origin of images to list. */
                image_origin?: components["schemas"]["ResourceOrigin"] | null;
                /** @description The categories of image to include. */
                categories?: components["schemas"]["ImageCategory"][] | null;
                /** @description Whether to list intermediate images. */
                is_intermediate?: boolean | null;
                /** @description The board id to filter by. Use 'none' to find images without a board. */
                board_id?: string | null;
                /** @description The page offset */
                offset?: number;
                /** @description The number of images per page */
                limit?: number;
                /** @description The order of sort */
                order_dir?: components["schemas"]["SQLiteDirection"];
                /** @description Whether to sort by starred images first */
                starred_first?: boolean;
                /** @description The term to search for */
                search_term?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OffsetPaginatedResults_ImageDTO_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_image_upload_entry: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_create_image_upload_entry"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImageUploadEntry"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_image_dto: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of image to get */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImageDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_image: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of the image to delete */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_image: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of the image to update */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ImageRecordChanges"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImageDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_intermediates_count: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": number;
                };
            };
        };
    };
    clear_intermediates: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": number;
                };
            };
        };
    };
    get_image_metadata: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of image to get */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MetadataField"] | null;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_image_workflow: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of image whose workflow to get */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WorkflowAndGraphResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_image_full: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of full-resolution image file to get */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Return the full-resolution image */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "image/png": unknown;
                };
            };
            /** @description Image not found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_image_full_head: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of full-resolution image file to get */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Return the full-resolution image */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "image/png": unknown;
                };
            };
            /** @description Image not found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_image_thumbnail: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of thumbnail image file to get */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Return the image thumbnail */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "image/webp": unknown;
                };
            };
            /** @description Image not found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_image_urls: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The name of the image whose URL to get */
                image_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImageUrlsDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_images_from_list: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_delete_images_from_list"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DeleteImagesFromListResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    star_images_in_list: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_star_images_in_list"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImagesUpdatedFromListResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    unstar_images_in_list: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_unstar_images_in_list"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImagesUpdatedFromListResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    download_images_from_list: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: {
            content: {
                "application/json": components["schemas"]["Body_download_images_from_list"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ImagesDownloaded"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_bulk_download_item: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The bulk_download_item_name of the bulk download item to get */
                bulk_download_item_name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Return the complete bulk download item */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/zip": unknown;
                };
            };
            /** @description Image not found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_boards: {
        parameters: {
            query?: {
                /** @description The attribute to order by */
                order_by?: components["schemas"]["BoardRecordOrderBy"];
                /** @description The direction to order by */
                direction?: components["schemas"]["SQLiteDirection"];
                /** @description Whether to list all boards */
                all?: boolean | null;
                /** @description The page offset */
                offset?: number | null;
                /** @description The number of boards per page */
                limit?: number | null;
                /** @description Whether or not to include archived boards in list */
                include_archived?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OffsetPaginatedResults_BoardDTO_"] | components["schemas"]["BoardDTO"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_board: {
        parameters: {
            query: {
                /** @description The name of the board to create */
                board_name: string;
                /** @description Whether the board is private */
                is_private?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The board was created successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BoardDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_board: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The id of board to get */
                board_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BoardDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_board: {
        parameters: {
            query?: {
                /** @description Permanently delete all images on the board */
                include_images?: boolean | null;
            };
            header?: never;
            path: {
                /** @description The id of board to delete */
                board_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DeleteBoardResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_board: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The id of board to update */
                board_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["BoardChanges"];
            };
        };
        responses: {
            /** @description The board was updated successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BoardDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_all_board_image_names: {
        parameters: {
            query?: {
                /** @description The categories of image to include. */
                categories?: components["schemas"]["ImageCategory"][] | null;
                /** @description Whether to list intermediate images. */
                is_intermediate?: boolean | null;
            };
            header?: never;
            path: {
                /** @description The id of the board */
                board_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": string[];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    add_image_to_board: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_add_image_to_board"];
            };
        };
        responses: {
            /** @description The image was added to a board successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    remove_image_from_board: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_remove_image_from_board"];
            };
        };
        responses: {
            /** @description The image was removed from the board successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    add_images_to_board: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_add_images_to_board"];
            };
        };
        responses: {
            /** @description Images were added to board successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AddImagesToBoardResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    remove_images_from_board: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_remove_images_from_board"];
            };
        };
        responses: {
            /** @description Images were removed from board successfully */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RemoveImagesFromBoardResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    app_version: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AppVersion"];
                };
            };
        };
    };
    get_app_deps: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AppDependencyVersions"];
                };
            };
        };
    };
    get_config: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AppConfig"];
                };
            };
        };
    };
    get_runtime_config: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["InvokeAIAppConfigWithSetFields"];
                };
            };
        };
    };
    get_log_level: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The operation was successful */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LogLevel"];
                };
            };
        };
    };
    set_log_level: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["LogLevel"];
            };
        };
        responses: {
            /** @description The operation was successful */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LogLevel"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    clear_invocation_cache: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The operation was successful */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    enable_invocation_cache: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The operation was successful */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    disable_invocation_cache: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The operation was successful */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    get_invocation_cache_status: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["InvocationCacheStatus"];
                };
            };
        };
    };
    enqueue_batch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_enqueue_batch"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["EnqueueBatchResult"];
                };
            };
            /** @description Created */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["EnqueueBatchResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_queue_items: {
        parameters: {
            query?: {
                /** @description The number of items to fetch */
                limit?: number;
                /** @description The status of items to fetch */
                status?: ("pending" | "in_progress" | "completed" | "failed" | "canceled") | null;
                /** @description The pagination cursor */
                cursor?: number | null;
                /** @description The pagination cursor priority */
                priority?: number;
            };
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CursorPaginatedResults_SessionQueueItemDTO_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    resume: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionProcessorStatus"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    pause: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionProcessorStatus"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_all_except_current: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CancelAllExceptCurrentResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_by_batch_ids: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_cancel_by_batch_ids"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CancelByBatchIDsResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_by_destination: {
        parameters: {
            query: {
                /** @description The destination to cancel all queue items for */
                destination: string;
            };
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CancelByDestinationResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    retry_items_by_id: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": number[];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RetryItemsResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    clear: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ClearResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    prune: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["PruneResult"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_current_queue_item: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionQueueItem"] | null | components["schemas"]["SessionQueueItem"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_next_queue_item: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionQueueItem"] | null | components["schemas"]["SessionQueueItem"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_queue_status: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionQueueAndProcessorStatus"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_batch_status: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
                /** @description The batch to get the status of */
                batch_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BatchStatus"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_queue_item: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
                /** @description The queue item to get */
                item_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionQueueItem"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_queue_item: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The queue id to perform this operation on */
                queue_id: string;
                /** @description The queue item to cancel */
                item_id: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionQueueItem"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    counts_by_destination: {
        parameters: {
            query: {
                /** @description The destination to query */
                destination: string;
            };
            header?: never;
            path: {
                /** @description The queue id to query */
                queue_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionQueueCountsByDestination"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_workflow: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The workflow to get */
                workflow_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WorkflowRecordWithThumbnailDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_workflow: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The workflow to delete */
                workflow_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_workflow: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_update_workflow"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WorkflowRecordDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_workflows: {
        parameters: {
            query?: {
                /** @description The page to get */
                page?: number;
                /** @description The number of workflows per page */
                per_page?: number | null;
                /** @description The attribute to order by */
                order_by?: components["schemas"]["WorkflowRecordOrderBy"];
                /** @description The direction to order by */
                direction?: components["schemas"]["SQLiteDirection"];
                /** @description The categories of workflow to get */
                categories?: components["schemas"]["WorkflowCategory"][] | null;
                /** @description The tags of workflow to get */
                tags?: string[] | null;
                /** @description The text to query by (matches name and description) */
                query?: string | null;
                /** @description Whether to include/exclude recent workflows */
                has_been_opened?: boolean | null;
                /** @description Whether to include/exclude published workflows */
                is_published?: boolean | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["PaginatedResults_WorkflowRecordListItemWithThumbnailDTO_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_workflow: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Body_create_workflow"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WorkflowRecordDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_workflow_thumbnail: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The id of the workflow thumbnail to get */
                workflow_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The workflow thumbnail was fetched successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description The workflow thumbnail could not be found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    set_workflow_thumbnail: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The workflow to update */
                workflow_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_set_workflow_thumbnail"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WorkflowRecordDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_workflow_thumbnail: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The workflow to update */
                workflow_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WorkflowRecordDTO"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_counts_by_tag: {
        parameters: {
            query: {
                /** @description The tags to get counts for */
                tags: string[];
                /** @description The categories to include */
                categories?: components["schemas"]["WorkflowCategory"][] | null;
                /** @description Whether to include/exclude recent workflows */
                has_been_opened?: boolean | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: number;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    counts_by_category: {
        parameters: {
            query: {
                /** @description The categories to include */
                categories: components["schemas"]["WorkflowCategory"][];
                /** @description Whether to include/exclude recent workflows */
                has_been_opened?: boolean | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: number;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_opened_at: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The workflow to update */
                workflow_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_style_preset: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The style preset to get */
                style_preset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StylePresetRecordWithImage"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_style_preset: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The style preset to delete */
                style_preset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_style_preset: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The id of the style preset to update */
                style_preset_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_update_style_preset"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StylePresetRecordWithImage"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_style_presets: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StylePresetRecordWithImage"][];
                };
            };
        };
    };
    create_style_preset: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_create_style_preset"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["StylePresetRecordWithImage"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_style_preset_image: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                /** @description The id of the style preset image to get */
                style_preset_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description The style preset image was fetched successfully */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Bad request */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description The style preset image could not be found */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    export_style_presets: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description A CSV file with the requested data. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                    "text/csv": unknown;
                };
            };
        };
    };
    import_style_presets: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_import_style_presets"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
