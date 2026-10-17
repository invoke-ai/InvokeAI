-- The schema of a mysql database at the portable-migration cutover (2026_10_01_add_anima_variant): every table
-- as 8.4.11 rendered it (SHOW CREATE TABLE) after creating it from the schema metadata of that time.
-- Frozen: test_portable_migrations.py migrates a database created from it and compares the result with
-- one created from today's metadata. Do not edit or regenerate it.

CREATE TABLE `app_settings` (
  `key` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `value` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `applied_migrations` (
  `migration_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `legacy_version` bigint DEFAULT NULL,
  `migrated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`migration_id`),
  UNIQUE KEY `uq_applied_migrations_legacy_version` (`legacy_version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_index_vocab_terms` (
  `term` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_as_ci NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`term`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_subfolder_move_jobs` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `state` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `error_message` longtext COLLATE utf8mb4_0900_bin,
  PRIMARY KEY (`id`),
  CONSTRAINT `ck_image_subfolder_move_jobs_state` CHECK ((`state` in (_utf8mb4'planned',_utf8mb4'moving',_utf8mb4'moved',_utf8mb4'committed',_utf8mb4'error')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `images` (
  `image_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `image_origin` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `image_category` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `width` bigint NOT NULL,
  `height` bigint NOT NULL,
  `session_id` longtext COLLATE utf8mb4_0900_bin,
  `node_id` longtext COLLATE utf8mb4_0900_bin,
  `metadata` longtext COLLATE utf8mb4_0900_bin,
  `is_intermediate` tinyint(1) DEFAULT '0',
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `deleted_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `starred` tinyint(1) DEFAULT '0',
  `has_workflow` tinyint(1) DEFAULT '0',
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin DEFAULT (_utf8mb4'system'),
  `image_subfolder` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4''),
  `project_id` varchar(255) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `file_size_bytes` bigint DEFAULT NULL,
  PRIMARY KEY (`image_name`),
  KEY `idx_images_created_at` (`created_at`),
  KEY `idx_images_unmeasured_intermediates` (`is_intermediate`,`file_size_bytes`,`created_at`),
  KEY `idx_images_image_origin` (`image_origin`),
  KEY `idx_images_user_id` (`user_id`),
  KEY `idx_images_image_category` (`image_category`),
  KEY `idx_images_intermediate_scope` (`is_intermediate`,`user_id`,`project_id`),
  KEY `idx_images_starred` (`starred`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `intermediates_browser_holds` (
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `lease_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `media_kind` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `media_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `expires_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`user_id`,`lease_id`,`media_kind`,`media_name`),
  KEY `idx_intermediates_browser_holds_expires_at` (`expires_at`),
  KEY `idx_intermediates_browser_holds_media` (`media_kind`,`media_name`,`expires_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `media_references` (
  `owner_kind` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `owner_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `media_kind` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `media_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`owner_kind`,`user_id`,`owner_id`,`media_kind`,`media_name`),
  KEY `idx_media_references_media` (`media_kind`,`media_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `migrations` (
  `version` bigint NOT NULL,
  `migrated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `models` (
  `id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `hash` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.hash')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.hash')) end)) VIRTUAL NOT NULL,
  `base` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.base')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.base')) end)) VIRTUAL NOT NULL,
  `type` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.type')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.type')) end)) VIRTUAL NOT NULL,
  `path` varchar(768) COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.path')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.path')) end)) VIRTUAL NOT NULL,
  `format` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.format')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.format')) end)) VIRTUAL NOT NULL,
  `name` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.name')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.name')) end)) VIRTUAL NOT NULL,
  `description` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.description')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.description')) end)) VIRTUAL,
  `source` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.source')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.source')) end)) VIRTUAL NOT NULL,
  `source_type` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.source_type')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.source_type')) end)) VIRTUAL NOT NULL,
  `source_api_response` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.source_api_response')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.source_api_response')) end)) VIRTUAL,
  `trigger_phrases` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`config`,_utf8mb4'$.trigger_phrases')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.trigger_phrases')) end)) VIRTUAL,
  `file_size` bigint GENERATED ALWAYS AS (cast((case when (json_type(json_extract(`config`,_utf8mb4'$.file_size')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`config`,_utf8mb4'$.file_size')) end) as signed)) VIRTUAL NOT NULL,
  `config` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_models_path` (`path`),
  KEY `type_index` (`type`(255)),
  KEY `base_index` (`base`(255)),
  KEY `name_index` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `orphaned_projects_2026_08_06` (
  `project_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `name` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `data` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `revision` bigint NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `quarantined_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`user_id`,`project_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `session_queue` (
  `item_id` bigint NOT NULL AUTO_INCREMENT,
  `batch_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `queue_id` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `session_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `field_values` longtext COLLATE utf8mb4_0900_bin,
  `session` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `status` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'pending'),
  `priority` bigint NOT NULL DEFAULT '0',
  `error_traceback` longtext COLLATE utf8mb4_0900_bin,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `started_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `completed_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `workflow` longtext COLLATE utf8mb4_0900_bin,
  `error_type` longtext COLLATE utf8mb4_0900_bin,
  `error_message` longtext COLLATE utf8mb4_0900_bin,
  `origin` longtext COLLATE utf8mb4_0900_bin,
  `destination` longtext COLLATE utf8mb4_0900_bin,
  `retried_from_item_id` bigint DEFAULT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin DEFAULT (_utf8mb4'system'),
  `status_sequence` bigint DEFAULT '0',
  `device` longtext COLLATE utf8mb4_0900_bin,
  `workflow_call_id` varchar(255) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `parent_item_id` bigint DEFAULT NULL,
  `parent_session_id` varchar(255) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `root_item_id` bigint DEFAULT NULL,
  `workflow_call_depth` bigint DEFAULT NULL,
  `project_id` longtext COLLATE utf8mb4_0900_bin,
  `session_revision` bigint NOT NULL DEFAULT '0',
  PRIMARY KEY (`item_id`),
  UNIQUE KEY `uq_session_queue_session_id` (`session_id`),
  KEY `idx_session_queue_parent_session_id` (`parent_session_id`),
  KEY `idx_session_queue_batch_id` (`batch_id`),
  KEY `idx_session_queue_workflow_call_depth` (`workflow_call_depth`),
  KEY `idx_session_queue_root_item_id` (`root_item_id`),
  KEY `idx_session_queue_parent_item_id` (`parent_item_id`),
  KEY `idx_session_queue_workflow_call_id` (`workflow_call_id`),
  KEY `idx_session_queue_round_robin_pending` (`status`,`user_id`,`priority` DESC,`item_id`),
  KEY `idx_session_queue_created_priority` (`priority`),
  KEY `idx_session_queue_user_started_at` (`user_id`,`started_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `style_presets` (
  `id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `name` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `preset_data` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `type` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'user'),
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin DEFAULT (_utf8mb4'system'),
  `is_public` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `idx_style_presets_user_id` (`user_id`),
  KEY `idx_style_presets_is_public` (`is_public`),
  KEY `idx_style_presets_name` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `system_prompts` (
  `id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `name` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `content` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'system'),
  `is_public` tinyint(1) NOT NULL DEFAULT '0',
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `max_tokens` bigint DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_system_prompts_user_id` (`user_id`),
  KEY `idx_system_prompts_name` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `users` (
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `email` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `display_name` longtext COLLATE utf8mb4_0900_bin,
  `password_hash` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `is_admin` tinyint(1) NOT NULL DEFAULT '0',
  `is_active` tinyint(1) NOT NULL DEFAULT '1',
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `last_login_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `token_epoch` bigint NOT NULL DEFAULT '0',
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uq_users_email` (`email`),
  KEY `idx_users_is_admin` (`is_admin`),
  KEY `idx_users_is_active` (`is_active`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `videos` (
  `video_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `video_origin` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `video_category` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `width` bigint NOT NULL,
  `height` bigint NOT NULL,
  `duration` double NOT NULL DEFAULT (0.0),
  `fps` double DEFAULT NULL,
  `session_id` longtext COLLATE utf8mb4_0900_bin,
  `node_id` longtext COLLATE utf8mb4_0900_bin,
  `metadata` longtext COLLATE utf8mb4_0900_bin,
  `is_intermediate` tinyint(1) DEFAULT '0',
  `starred` tinyint(1) DEFAULT '0',
  `has_workflow` tinyint(1) DEFAULT '0',
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'system'),
  `video_subfolder` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4''),
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `deleted_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `project_id` varchar(255) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `file_size_bytes` bigint DEFAULT NULL,
  PRIMARY KEY (`video_name`),
  KEY `idx_videos_video_category` (`video_category`),
  KEY `idx_videos_video_origin` (`video_origin`),
  KEY `idx_videos_intermediate_scope` (`is_intermediate`,`user_id`,`project_id`),
  KEY `idx_videos_starred` (`starred`),
  KEY `idx_videos_unmeasured_intermediates` (`is_intermediate`,`file_size_bytes`,`created_at`),
  KEY `idx_videos_user_id` (`user_id`),
  KEY `idx_videos_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `workflow_library` (
  `workflow_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `workflow` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `category` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`workflow`,_utf8mb4'$.meta.category')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`workflow`,_utf8mb4'$.meta.category')) end)) VIRTUAL NOT NULL,
  `name` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`workflow`,_utf8mb4'$.name')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`workflow`,_utf8mb4'$.name')) end)) VIRTUAL NOT NULL,
  `description` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`workflow`,_utf8mb4'$.description')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`workflow`,_utf8mb4'$.description')) end)) VIRTUAL NOT NULL,
  `tags` longtext COLLATE utf8mb4_0900_bin GENERATED ALWAYS AS ((case when (json_type(json_extract(`workflow`,_utf8mb4'$.tags')) = _utf8mb4'NULL') then NULL else json_unquote(json_extract(`workflow`,_utf8mb4'$.tags')) end)) VIRTUAL,
  `opened_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin DEFAULT (_utf8mb4'system'),
  `is_public` tinyint(1) NOT NULL DEFAULT '0',
  `last_run_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `revision` bigint NOT NULL DEFAULT '1',
  PRIMARY KEY (`workflow_id`),
  KEY `idx_workflow_library_is_public` (`is_public`),
  KEY `idx_workflow_library_opened_at` (`opened_at`),
  KEY `idx_workflow_library_created_at` (`created_at`),
  KEY `idx_workflow_library_updated_at` (`updated_at`),
  KEY `idx_workflow_library_user_id` (`user_id`),
  KEY `idx_workflow_library_category` (`category`(255)),
  KEY `idx_workflow_library_description` (`description`(255)),
  KEY `idx_workflow_library_name` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `boards` (
  `board_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `board_name` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `cover_image_name` varchar(255) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `deleted_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `archived` tinyint(1) DEFAULT '0',
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin DEFAULT (_utf8mb4'system'),
  `is_public` tinyint(1) NOT NULL DEFAULT '0',
  `board_visibility` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'private'),
  PRIMARY KEY (`board_id`),
  KEY `fk_boards_cover_image_name_images` (`cover_image_name`),
  KEY `idx_boards_board_visibility` (`board_visibility`),
  KEY `idx_boards_created_at` (`created_at`),
  KEY `idx_boards_user_id` (`user_id`),
  KEY `idx_boards_is_public` (`is_public`),
  CONSTRAINT `fk_boards_cover_image_name_images` FOREIGN KEY (`cover_image_name`) REFERENCES `images` (`image_name`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `client_state` (
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `key` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `value` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`user_id`,`key`),
  CONSTRAINT `fk_client_state_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `fonts` (
  `id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `owner_id` varchar(64) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `scope` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `source` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `filename` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `storage_path` longtext COLLATE utf8mb4_0900_bin,
  `source_path` varchar(768) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `family` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `label` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `style` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `weight` bigint NOT NULL,
  `content_hash` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `byte_size` bigint NOT NULL,
  `axes_json` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'[]'),
  `instances_json` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'[]'),
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_fonts_directory_source_path` (`source_path`),
  KEY `fk_fonts_owner_id_users` (`owner_id`),
  KEY `idx_fonts_uploaded_hash` (`source`,`scope`,`owner_id`,`content_hash`),
  CONSTRAINT `fk_fonts_owner_id_users` FOREIGN KEY (`owner_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE,
  CONSTRAINT `ck_fonts_byte_size` CHECK ((`byte_size` > 0)),
  CONSTRAINT `ck_fonts_content_hash` CHECK ((length(`content_hash`) = 64)),
  CONSTRAINT `ck_fonts_scope` CHECK ((`scope` in (_utf8mb4'private',_utf8mb4'shared'))),
  CONSTRAINT `ck_fonts_source` CHECK ((`source` in (_utf8mb4'uploaded',_utf8mb4'directory'))),
  CONSTRAINT `ck_fonts_source_fields` CHECK ((((`source` = _utf8mb4'directory') and (`owner_id` is null) and (`scope` = _utf8mb4'shared') and (`storage_path` is null) and (`source_path` is not null)) or ((`source` = _utf8mb4'uploaded') and (`storage_path` is not null) and (`source_path` is null) and (((`scope` = _utf8mb4'private') and (`owner_id` is not null)) or ((`scope` = _utf8mb4'shared') and (`owner_id` is null)))))),
  CONSTRAINT `ck_fonts_weight` CHECK (((`weight` >= 1) and (`weight` <= 1000)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_embeddings` (
  `image_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `model_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `dim` bigint NOT NULL,
  `embedding` longblob NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `encoding` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'float32'),
  PRIMARY KEY (`image_name`,`model_id`),
  KEY `idx_image_embeddings_model_id` (`model_id`),
  CONSTRAINT `fk_image_embeddings_image_name_images` FOREIGN KEY (`image_name`) REFERENCES `images` (`image_name`) ON DELETE CASCADE,
  CONSTRAINT `ck_image_embeddings_encoding` CHECK ((`encoding` in (_utf8mb4'float32',_utf8mb4'float16')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_projections` (
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `model_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `scope_hash` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `params` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `point_count` bigint NOT NULL,
  `image_names` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `coords` longblob NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `item_kinds` longtext COLLATE utf8mb4_0900_bin,
  PRIMARY KEY (`user_id`,`model_id`),
  CONSTRAINT `fk_image_projections_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_subfolder_move_items` (
  `job_id` bigint NOT NULL,
  `image_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `old_subfolder` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `new_subfolder` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `is_intermediate` tinyint(1) NOT NULL DEFAULT '0',
  `old_path` longtext COLLATE utf8mb4_0900_bin,
  `new_path` longtext COLLATE utf8mb4_0900_bin,
  `old_thumbnail_path` longtext COLLATE utf8mb4_0900_bin,
  `new_thumbnail_path` longtext COLLATE utf8mb4_0900_bin,
  `state` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `error_message` longtext COLLATE utf8mb4_0900_bin,
  PRIMARY KEY (`job_id`,`image_name`),
  KEY `idx_image_subfolder_move_items_image_name` (`image_name`),
  KEY `idx_image_subfolder_move_items_job_state` (`job_id`,`state`),
  CONSTRAINT `fk_image_subfolder_move_items_image_name_images` FOREIGN KEY (`image_name`) REFERENCES `images` (`image_name`) ON DELETE CASCADE,
  CONSTRAINT `fk_image_subfolder_move_items_job_id_image_subfolder_move_jobs` FOREIGN KEY (`job_id`) REFERENCES `image_subfolder_move_jobs` (`id`),
  CONSTRAINT `ck_image_subfolder_move_items_state` CHECK ((`state` in (_utf8mb4'planned',_utf8mb4'moved',_utf8mb4'committed',_utf8mb4'error')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `model_relationships` (
  `model_key_1` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `model_key_2` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`model_key_1`,`model_key_2`),
  KEY `keyx_model_relationships_model_key_2` (`model_key_2`),
  CONSTRAINT `fk_model_relationships_model_key_1_models` FOREIGN KEY (`model_key_1`) REFERENCES `models` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_model_relationships_model_key_2_models` FOREIGN KEY (`model_key_2`) REFERENCES `models` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `session_queue_enqueue_receipts` (
  `queue_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `idempotency_key` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `payload_hash` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `batch_id` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `requested` bigint NOT NULL,
  `enqueued` bigint NOT NULL,
  `priority` bigint NOT NULL,
  `item_ids` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `byte_size` bigint NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `acknowledged_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  PRIMARY KEY (`queue_id`,`user_id`,`idempotency_key`),
  KEY `idx_session_queue_enqueue_receipts_owner_ack` (`user_id`,`acknowledged_at`,`byte_size`),
  CONSTRAINT `fk_session_queue_enqueue_receipts_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `user_invitations` (
  `invitation_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `email` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `invited_by` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `invitation_code` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `is_admin` tinyint(1) NOT NULL DEFAULT '0',
  `expires_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `used_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`invitation_id`),
  UNIQUE KEY `uq_user_invitations_invitation_code` (`invitation_code`),
  KEY `fk_user_invitations_invited_by_users` (`invited_by`),
  KEY `idx_user_invitations_expires_at` (`expires_at`),
  KEY `idx_user_invitations_email` (`email`),
  CONSTRAINT `fk_user_invitations_invited_by_users` FOREIGN KEY (`invited_by`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `user_sessions` (
  `session_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `token_hash` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `expires_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `last_activity_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`session_id`),
  KEY `idx_user_sessions_token_hash` (`token_hash`),
  KEY `idx_user_sessions_expires_at` (`expires_at`),
  KEY `idx_user_sessions_user_id` (`user_id`),
  CONSTRAINT `fk_user_sessions_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `video_embeddings` (
  `video_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `model_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `dim` bigint NOT NULL,
  `embedding` longblob NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `encoding` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'float32'),
  PRIMARY KEY (`video_name`,`model_id`),
  KEY `idx_video_embeddings_model_id` (`model_id`),
  CONSTRAINT `fk_video_embeddings_video_name_videos` FOREIGN KEY (`video_name`) REFERENCES `videos` (`video_name`) ON DELETE CASCADE,
  CONSTRAINT `ck_video_embeddings_encoding` CHECK ((`encoding` in (_utf8mb4'float32',_utf8mb4'float16')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `wildcards` (
  `id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `values_json` longtext COLLATE utf8mb4_0900_bin NOT NULL DEFAULT (_utf8mb4'[]'),
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_wildcards_user_id_name` (`user_id`,`name`),
  CONSTRAINT `fk_wildcards_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `board_images` (
  `board_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `image_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `deleted_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  PRIMARY KEY (`image_name`),
  KEY `idx_board_images_board_id_created_at` (`board_id`,`created_at`),
  CONSTRAINT `fk_board_images_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_board_images_image_name_images` FOREIGN KEY (`image_name`) REFERENCES `images` (`image_name`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `board_videos` (
  `board_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `video_name` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `deleted_at` varchar(32) COLLATE utf8mb4_0900_bin DEFAULT NULL,
  PRIMARY KEY (`video_name`),
  KEY `idx_board_videos_board_id_created_at` (`board_id`,`created_at`),
  CONSTRAINT `fk_board_videos_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_board_videos_video_name_videos` FOREIGN KEY (`video_name`) REFERENCES `videos` (`video_name`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `projects` (
  `project_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `name` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `data` longtext COLLATE utf8mb4_0900_bin NOT NULL,
  `board_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `revision` bigint NOT NULL DEFAULT '1',
  `created_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `updated_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  `minimum_canvas_schema_version` bigint NOT NULL DEFAULT '2',
  PRIMARY KEY (`user_id`,`project_id`),
  UNIQUE KEY `uq_projects_board_id` (`board_id`),
  CONSTRAINT `fk_projects_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`) ON DELETE RESTRICT,
  CONSTRAINT `fk_projects_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE,
  CONSTRAINT `ck_projects_minimum_canvas_schema_version` CHECK ((`minimum_canvas_schema_version` >= 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `shared_boards` (
  `board_id` varchar(255) COLLATE utf8mb4_0900_bin NOT NULL,
  `user_id` varchar(64) COLLATE utf8mb4_0900_bin NOT NULL,
  `can_edit` tinyint(1) NOT NULL DEFAULT '0',
  `shared_at` varchar(32) COLLATE utf8mb4_0900_bin NOT NULL,
  PRIMARY KEY (`board_id`,`user_id`),
  KEY `idx_shared_boards_user_id` (`user_id`),
  CONSTRAINT `fk_shared_boards_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_shared_boards_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin ROW_FORMAT=DYNAMIC;
