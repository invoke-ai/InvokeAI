-- The schema of a mariadb database at the portable-migration cutover (2026_10_01_add_anima_variant): every table
-- as 10.11.19-MariaDB-ubu2204 rendered it (SHOW CREATE TABLE) after creating it from the schema metadata of that time.
-- Frozen: test_portable_migrations.py migrates a database created from it and compares the result with
-- one created from today's metadata. Do not edit or regenerate it.

CREATE TABLE `app_settings` (
  `key` varchar(255) NOT NULL,
  `value` longtext NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  PRIMARY KEY (`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `applied_migrations` (
  `migration_id` varchar(255) NOT NULL,
  `legacy_version` bigint(20) DEFAULT NULL,
  `migrated_at` varchar(32) NOT NULL,
  PRIMARY KEY (`migration_id`),
  UNIQUE KEY `uq_applied_migrations_legacy_version` (`legacy_version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_index_vocab_terms` (
  `term` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_uca1400_nopad_as_ci NOT NULL,
  `created_at` varchar(32) NOT NULL,
  PRIMARY KEY (`term`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_subfolder_move_jobs` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `state` longtext NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `error_message` longtext DEFAULT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `ck_image_subfolder_move_jobs_state` CHECK (`state` in ('planned','moving','moved','committed','error'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `images` (
  `image_name` varchar(255) NOT NULL,
  `image_origin` varchar(32) NOT NULL,
  `image_category` varchar(32) NOT NULL,
  `width` bigint(20) NOT NULL,
  `height` bigint(20) NOT NULL,
  `session_id` longtext DEFAULT NULL,
  `node_id` longtext DEFAULT NULL,
  `metadata` longtext DEFAULT NULL,
  `is_intermediate` tinyint(1) DEFAULT 0,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `deleted_at` varchar(32) DEFAULT NULL,
  `starred` tinyint(1) DEFAULT 0,
  `has_workflow` tinyint(1) DEFAULT 0,
  `user_id` varchar(64) DEFAULT 'system',
  `image_subfolder` longtext NOT NULL DEFAULT '',
  `project_id` varchar(255) DEFAULT NULL,
  `file_size_bytes` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`image_name`),
  KEY `idx_images_created_at` (`created_at`),
  KEY `idx_images_unmeasured_intermediates` (`is_intermediate`,`file_size_bytes`,`created_at`),
  KEY `idx_images_image_origin` (`image_origin`),
  KEY `idx_images_user_id` (`user_id`),
  KEY `idx_images_image_category` (`image_category`),
  KEY `idx_images_intermediate_scope` (`is_intermediate`,`user_id`,`project_id`),
  KEY `idx_images_starred` (`starred`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `intermediates_browser_holds` (
  `user_id` varchar(64) NOT NULL,
  `lease_id` varchar(255) NOT NULL,
  `media_kind` varchar(32) NOT NULL,
  `media_name` varchar(255) NOT NULL,
  `expires_at` varchar(32) NOT NULL,
  PRIMARY KEY (`user_id`,`lease_id`,`media_kind`,`media_name`),
  KEY `idx_intermediates_browser_holds_expires_at` (`expires_at`),
  KEY `idx_intermediates_browser_holds_media` (`media_kind`,`media_name`,`expires_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `media_references` (
  `owner_kind` varchar(32) NOT NULL,
  `user_id` varchar(64) NOT NULL,
  `owner_id` varchar(255) NOT NULL,
  `media_kind` varchar(32) NOT NULL,
  `media_name` varchar(255) NOT NULL,
  PRIMARY KEY (`owner_kind`,`user_id`,`owner_id`,`media_kind`,`media_name`),
  KEY `idx_media_references_media` (`media_kind`,`media_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `migrations` (
  `version` bigint(20) NOT NULL,
  `migrated_at` varchar(32) NOT NULL,
  PRIMARY KEY (`version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `models` (
  `id` varchar(255) NOT NULL,
  `hash` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.hash')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.hash')) end) VIRTUAL,
  `base` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.base')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.base')) end) VIRTUAL,
  `type` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.type')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.type')) end) VIRTUAL,
  `path` varchar(768) GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.path')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.path')) end) VIRTUAL,
  `format` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.format')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.format')) end) VIRTUAL,
  `name` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.name')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.name')) end) VIRTUAL,
  `description` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.description')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.description')) end) VIRTUAL,
  `source` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.source')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.source')) end) VIRTUAL,
  `source_type` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.source_type')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.source_type')) end) VIRTUAL,
  `source_api_response` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.source_api_response')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.source_api_response')) end) VIRTUAL,
  `trigger_phrases` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`config`,'$.trigger_phrases')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.trigger_phrases')) end) VIRTUAL,
  `file_size` bigint(20) GENERATED ALWAYS AS (cast(case when json_type(json_extract(`config`,'$.file_size')) = 'NULL' then NULL else json_unquote(json_extract(`config`,'$.file_size')) end as signed)) VIRTUAL,
  `config` longtext NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_models_path` (`path`),
  KEY `type_index` (`type`(255)),
  KEY `base_index` (`base`(255)),
  KEY `name_index` (`name`(255)),
  CONSTRAINT `ck_models_hash_not_null` CHECK (`hash` is not null),
  CONSTRAINT `ck_models_base_not_null` CHECK (`base` is not null),
  CONSTRAINT `ck_models_type_not_null` CHECK (`type` is not null),
  CONSTRAINT `ck_models_path_not_null` CHECK (`path` is not null),
  CONSTRAINT `ck_models_format_not_null` CHECK (`format` is not null),
  CONSTRAINT `ck_models_name_not_null` CHECK (`name` is not null),
  CONSTRAINT `ck_models_source_not_null` CHECK (`source` is not null),
  CONSTRAINT `ck_models_source_type_not_null` CHECK (`source_type` is not null),
  CONSTRAINT `ck_models_file_size_not_null` CHECK (`file_size` is not null)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `orphaned_projects_2026_08_06` (
  `project_id` varchar(255) NOT NULL,
  `user_id` varchar(64) NOT NULL,
  `name` longtext NOT NULL,
  `data` longtext NOT NULL,
  `revision` bigint(20) NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `quarantined_at` varchar(32) NOT NULL,
  PRIMARY KEY (`user_id`,`project_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `session_queue` (
  `item_id` bigint(20) NOT NULL AUTO_INCREMENT,
  `batch_id` varchar(255) NOT NULL,
  `queue_id` longtext NOT NULL,
  `session_id` varchar(255) NOT NULL,
  `field_values` longtext DEFAULT NULL,
  `session` longtext NOT NULL,
  `status` varchar(32) NOT NULL DEFAULT 'pending',
  `priority` bigint(20) NOT NULL DEFAULT 0,
  `error_traceback` longtext DEFAULT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `started_at` varchar(32) DEFAULT NULL,
  `completed_at` varchar(32) DEFAULT NULL,
  `workflow` longtext DEFAULT NULL,
  `error_type` longtext DEFAULT NULL,
  `error_message` longtext DEFAULT NULL,
  `origin` longtext DEFAULT NULL,
  `destination` longtext DEFAULT NULL,
  `retried_from_item_id` bigint(20) DEFAULT NULL,
  `user_id` varchar(64) DEFAULT 'system',
  `status_sequence` bigint(20) DEFAULT 0,
  `device` longtext DEFAULT NULL,
  `workflow_call_id` varchar(255) DEFAULT NULL,
  `parent_item_id` bigint(20) DEFAULT NULL,
  `parent_session_id` varchar(255) DEFAULT NULL,
  `root_item_id` bigint(20) DEFAULT NULL,
  `workflow_call_depth` bigint(20) DEFAULT NULL,
  `project_id` longtext DEFAULT NULL,
  `session_revision` bigint(20) NOT NULL DEFAULT 0,
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `style_presets` (
  `id` varchar(255) NOT NULL,
  `name` longtext NOT NULL,
  `preset_data` longtext NOT NULL,
  `type` longtext NOT NULL DEFAULT 'user',
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `user_id` varchar(64) DEFAULT 'system',
  `is_public` tinyint(1) NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  KEY `idx_style_presets_user_id` (`user_id`),
  KEY `idx_style_presets_is_public` (`is_public`),
  KEY `idx_style_presets_name` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `system_prompts` (
  `id` varchar(255) NOT NULL,
  `name` longtext NOT NULL,
  `content` longtext NOT NULL,
  `user_id` varchar(64) NOT NULL DEFAULT 'system',
  `is_public` tinyint(1) NOT NULL DEFAULT 0,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `max_tokens` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_system_prompts_user_id` (`user_id`),
  KEY `idx_system_prompts_name` (`name`(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `users` (
  `user_id` varchar(64) NOT NULL,
  `email` varchar(255) NOT NULL,
  `display_name` longtext DEFAULT NULL,
  `password_hash` longtext NOT NULL,
  `is_admin` tinyint(1) NOT NULL DEFAULT 0,
  `is_active` tinyint(1) NOT NULL DEFAULT 1,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `last_login_at` varchar(32) DEFAULT NULL,
  `token_epoch` bigint(20) NOT NULL DEFAULT 0,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uq_users_email` (`email`),
  KEY `idx_users_is_admin` (`is_admin`),
  KEY `idx_users_is_active` (`is_active`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `videos` (
  `video_name` varchar(255) NOT NULL,
  `video_origin` varchar(32) NOT NULL,
  `video_category` varchar(32) NOT NULL,
  `width` bigint(20) NOT NULL,
  `height` bigint(20) NOT NULL,
  `duration` double NOT NULL DEFAULT 0,
  `fps` double DEFAULT NULL,
  `session_id` longtext DEFAULT NULL,
  `node_id` longtext DEFAULT NULL,
  `metadata` longtext DEFAULT NULL,
  `is_intermediate` tinyint(1) DEFAULT 0,
  `starred` tinyint(1) DEFAULT 0,
  `has_workflow` tinyint(1) DEFAULT 0,
  `user_id` varchar(64) NOT NULL DEFAULT 'system',
  `video_subfolder` longtext NOT NULL DEFAULT '',
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `deleted_at` varchar(32) DEFAULT NULL,
  `project_id` varchar(255) DEFAULT NULL,
  `file_size_bytes` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`video_name`),
  KEY `idx_videos_video_category` (`video_category`),
  KEY `idx_videos_video_origin` (`video_origin`),
  KEY `idx_videos_intermediate_scope` (`is_intermediate`,`user_id`,`project_id`),
  KEY `idx_videos_starred` (`starred`),
  KEY `idx_videos_unmeasured_intermediates` (`is_intermediate`,`file_size_bytes`,`created_at`),
  KEY `idx_videos_user_id` (`user_id`),
  KEY `idx_videos_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `workflow_library` (
  `workflow_id` varchar(255) NOT NULL,
  `workflow` longtext NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `category` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`workflow`,'$.meta.category')) = 'NULL' then NULL else json_unquote(json_extract(`workflow`,'$.meta.category')) end) VIRTUAL,
  `name` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`workflow`,'$.name')) = 'NULL' then NULL else json_unquote(json_extract(`workflow`,'$.name')) end) VIRTUAL,
  `description` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`workflow`,'$.description')) = 'NULL' then NULL else json_unquote(json_extract(`workflow`,'$.description')) end) VIRTUAL,
  `tags` longtext GENERATED ALWAYS AS (case when json_type(json_extract(`workflow`,'$.tags')) = 'NULL' then NULL else json_unquote(json_extract(`workflow`,'$.tags')) end) VIRTUAL,
  `opened_at` varchar(32) DEFAULT NULL,
  `user_id` varchar(64) DEFAULT 'system',
  `is_public` tinyint(1) NOT NULL DEFAULT 0,
  `last_run_at` varchar(32) DEFAULT NULL,
  `revision` bigint(20) NOT NULL DEFAULT 1,
  PRIMARY KEY (`workflow_id`),
  KEY `idx_workflow_library_is_public` (`is_public`),
  KEY `idx_workflow_library_category` (`category`(255)),
  KEY `idx_workflow_library_description` (`description`(255)),
  KEY `idx_workflow_library_opened_at` (`opened_at`),
  KEY `idx_workflow_library_created_at` (`created_at`),
  KEY `idx_workflow_library_updated_at` (`updated_at`),
  KEY `idx_workflow_library_name` (`name`(255)),
  KEY `idx_workflow_library_user_id` (`user_id`),
  CONSTRAINT `ck_workflow_library_category_not_null` CHECK (`category` is not null),
  CONSTRAINT `ck_workflow_library_name_not_null` CHECK (`name` is not null),
  CONSTRAINT `ck_workflow_library_description_not_null` CHECK (`description` is not null)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `boards` (
  `board_id` varchar(255) NOT NULL,
  `board_name` longtext NOT NULL,
  `cover_image_name` varchar(255) DEFAULT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `deleted_at` varchar(32) DEFAULT NULL,
  `archived` tinyint(1) DEFAULT 0,
  `user_id` varchar(64) DEFAULT 'system',
  `is_public` tinyint(1) NOT NULL DEFAULT 0,
  `board_visibility` varchar(32) NOT NULL DEFAULT 'private',
  PRIMARY KEY (`board_id`),
  KEY `fk_boards_cover_image_name_images` (`cover_image_name`),
  KEY `idx_boards_board_visibility` (`board_visibility`),
  KEY `idx_boards_created_at` (`created_at`),
  KEY `idx_boards_user_id` (`user_id`),
  KEY `idx_boards_is_public` (`is_public`),
  CONSTRAINT `fk_boards_cover_image_name_images` FOREIGN KEY (`cover_image_name`) REFERENCES `images` (`image_name`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `client_state` (
  `user_id` varchar(64) NOT NULL,
  `key` varchar(255) NOT NULL,
  `value` longtext NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  PRIMARY KEY (`user_id`,`key`),
  CONSTRAINT `fk_client_state_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `fonts` (
  `id` varchar(255) NOT NULL,
  `owner_id` varchar(64) DEFAULT NULL,
  `scope` varchar(32) NOT NULL,
  `source` varchar(32) NOT NULL,
  `filename` longtext NOT NULL,
  `storage_path` longtext DEFAULT NULL,
  `source_path` varchar(768) DEFAULT NULL,
  `family` longtext NOT NULL,
  `label` longtext NOT NULL,
  `style` longtext NOT NULL,
  `weight` bigint(20) NOT NULL,
  `content_hash` varchar(64) NOT NULL,
  `byte_size` bigint(20) NOT NULL,
  `axes_json` longtext NOT NULL DEFAULT '[]',
  `instances_json` longtext NOT NULL DEFAULT '[]',
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_fonts_directory_source_path` (`source_path`),
  KEY `fk_fonts_owner_id_users` (`owner_id`),
  KEY `idx_fonts_uploaded_hash` (`source`,`scope`,`owner_id`,`content_hash`),
  CONSTRAINT `fk_fonts_owner_id_users` FOREIGN KEY (`owner_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE,
  CONSTRAINT `ck_fonts_scope` CHECK (`scope` in ('private','shared')),
  CONSTRAINT `ck_fonts_source` CHECK (`source` in ('uploaded','directory')),
  CONSTRAINT `ck_fonts_weight` CHECK (`weight` >= 1 and `weight` <= 1000),
  CONSTRAINT `ck_fonts_content_hash` CHECK (octet_length(`content_hash`) = 64),
  CONSTRAINT `ck_fonts_byte_size` CHECK (`byte_size` > 0),
  CONSTRAINT `ck_fonts_source_fields` CHECK (`source` = 'directory' and `owner_id` is null and `scope` = 'shared' and `storage_path` is null and `source_path` is not null or `source` = 'uploaded' and `storage_path` is not null and `source_path` is null and (`scope` = 'private' and `owner_id` is not null or `scope` = 'shared' and `owner_id` is null))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_embeddings` (
  `image_name` varchar(255) NOT NULL,
  `model_id` varchar(255) NOT NULL,
  `dim` bigint(20) NOT NULL,
  `embedding` longblob NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `encoding` longtext NOT NULL DEFAULT 'float32',
  PRIMARY KEY (`image_name`,`model_id`),
  KEY `idx_image_embeddings_model_id` (`model_id`),
  CONSTRAINT `fk_image_embeddings_image_name_images` FOREIGN KEY (`image_name`) REFERENCES `images` (`image_name`) ON DELETE CASCADE,
  CONSTRAINT `ck_image_embeddings_encoding` CHECK (`encoding` in ('float32','float16'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_projections` (
  `user_id` varchar(64) NOT NULL,
  `model_id` varchar(255) NOT NULL,
  `scope_hash` longtext NOT NULL,
  `params` longtext NOT NULL,
  `point_count` bigint(20) NOT NULL,
  `image_names` longtext NOT NULL,
  `coords` longblob NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `item_kinds` longtext DEFAULT NULL,
  PRIMARY KEY (`user_id`,`model_id`),
  CONSTRAINT `fk_image_projections_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `image_subfolder_move_items` (
  `job_id` bigint(20) NOT NULL,
  `image_name` varchar(255) NOT NULL,
  `old_subfolder` longtext NOT NULL,
  `new_subfolder` longtext NOT NULL,
  `is_intermediate` tinyint(1) NOT NULL DEFAULT 0,
  `old_path` longtext DEFAULT NULL,
  `new_path` longtext DEFAULT NULL,
  `old_thumbnail_path` longtext DEFAULT NULL,
  `new_thumbnail_path` longtext DEFAULT NULL,
  `state` varchar(32) NOT NULL,
  `error_message` longtext DEFAULT NULL,
  PRIMARY KEY (`job_id`,`image_name`),
  KEY `idx_image_subfolder_move_items_image_name` (`image_name`),
  KEY `idx_image_subfolder_move_items_job_state` (`job_id`,`state`),
  CONSTRAINT `fk_image_subfolder_move_items_image_name_images` FOREIGN KEY (`image_name`) REFERENCES `images` (`image_name`) ON DELETE CASCADE,
  CONSTRAINT `fk_image_subfolder_move_items_job_id_image_subfolder_move_jobs` FOREIGN KEY (`job_id`) REFERENCES `image_subfolder_move_jobs` (`id`),
  CONSTRAINT `ck_image_subfolder_move_items_state` CHECK (`state` in ('planned','moved','committed','error'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `model_relationships` (
  `model_key_1` varchar(255) NOT NULL,
  `model_key_2` varchar(255) NOT NULL,
  `created_at` varchar(32) NOT NULL,
  PRIMARY KEY (`model_key_1`,`model_key_2`),
  KEY `keyx_model_relationships_model_key_2` (`model_key_2`),
  CONSTRAINT `fk_model_relationships_model_key_1_models` FOREIGN KEY (`model_key_1`) REFERENCES `models` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_model_relationships_model_key_2_models` FOREIGN KEY (`model_key_2`) REFERENCES `models` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `session_queue_enqueue_receipts` (
  `queue_id` varchar(255) NOT NULL,
  `user_id` varchar(64) NOT NULL,
  `idempotency_key` varchar(255) NOT NULL,
  `payload_hash` longtext NOT NULL,
  `batch_id` longtext NOT NULL,
  `requested` bigint(20) NOT NULL,
  `enqueued` bigint(20) NOT NULL,
  `priority` bigint(20) NOT NULL,
  `item_ids` longtext NOT NULL,
  `byte_size` bigint(20) NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `acknowledged_at` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`queue_id`,`user_id`,`idempotency_key`),
  KEY `idx_session_queue_enqueue_receipts_owner_ack` (`user_id`,`acknowledged_at`,`byte_size`),
  CONSTRAINT `fk_session_queue_enqueue_receipts_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `user_invitations` (
  `invitation_id` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `invited_by` varchar(64) NOT NULL,
  `invitation_code` varchar(255) NOT NULL,
  `is_admin` tinyint(1) NOT NULL DEFAULT 0,
  `expires_at` varchar(32) NOT NULL,
  `used_at` varchar(32) DEFAULT NULL,
  `created_at` varchar(32) NOT NULL,
  PRIMARY KEY (`invitation_id`),
  UNIQUE KEY `uq_user_invitations_invitation_code` (`invitation_code`),
  KEY `fk_user_invitations_invited_by_users` (`invited_by`),
  KEY `idx_user_invitations_expires_at` (`expires_at`),
  KEY `idx_user_invitations_email` (`email`),
  CONSTRAINT `fk_user_invitations_invited_by_users` FOREIGN KEY (`invited_by`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `user_sessions` (
  `session_id` varchar(255) NOT NULL,
  `user_id` varchar(64) NOT NULL,
  `token_hash` varchar(255) NOT NULL,
  `expires_at` varchar(32) NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `last_activity_at` varchar(32) NOT NULL,
  PRIMARY KEY (`session_id`),
  KEY `idx_user_sessions_token_hash` (`token_hash`),
  KEY `idx_user_sessions_expires_at` (`expires_at`),
  KEY `idx_user_sessions_user_id` (`user_id`),
  CONSTRAINT `fk_user_sessions_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `video_embeddings` (
  `video_name` varchar(255) NOT NULL,
  `model_id` varchar(255) NOT NULL,
  `dim` bigint(20) NOT NULL,
  `embedding` longblob NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `encoding` longtext NOT NULL DEFAULT 'float32',
  PRIMARY KEY (`video_name`,`model_id`),
  KEY `idx_video_embeddings_model_id` (`model_id`),
  CONSTRAINT `fk_video_embeddings_video_name_videos` FOREIGN KEY (`video_name`) REFERENCES `videos` (`video_name`) ON DELETE CASCADE,
  CONSTRAINT `ck_video_embeddings_encoding` CHECK (`encoding` in ('float32','float16'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `wildcards` (
  `id` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `values_json` longtext NOT NULL DEFAULT '[]',
  `user_id` varchar(64) NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_wildcards_user_id_name` (`user_id`,`name`),
  CONSTRAINT `fk_wildcards_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `board_images` (
  `board_id` varchar(255) NOT NULL,
  `image_name` varchar(255) NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `deleted_at` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`image_name`),
  KEY `idx_board_images_board_id_created_at` (`board_id`,`created_at`),
  CONSTRAINT `fk_board_images_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_board_images_image_name_images` FOREIGN KEY (`image_name`) REFERENCES `images` (`image_name`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `board_videos` (
  `board_id` varchar(255) NOT NULL,
  `video_name` varchar(255) NOT NULL,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `deleted_at` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`video_name`),
  KEY `idx_board_videos_board_id_created_at` (`board_id`,`created_at`),
  CONSTRAINT `fk_board_videos_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_board_videos_video_name_videos` FOREIGN KEY (`video_name`) REFERENCES `videos` (`video_name`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `projects` (
  `project_id` varchar(255) NOT NULL,
  `user_id` varchar(64) NOT NULL,
  `name` longtext NOT NULL,
  `data` longtext NOT NULL,
  `board_id` varchar(255) NOT NULL,
  `revision` bigint(20) NOT NULL DEFAULT 1,
  `created_at` varchar(32) NOT NULL,
  `updated_at` varchar(32) NOT NULL,
  `minimum_canvas_schema_version` bigint(20) NOT NULL DEFAULT 2,
  PRIMARY KEY (`user_id`,`project_id`),
  UNIQUE KEY `uq_projects_board_id` (`board_id`),
  CONSTRAINT `fk_projects_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`),
  CONSTRAINT `fk_projects_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE,
  CONSTRAINT `ck_projects_minimum_canvas_schema_version` CHECK (`minimum_canvas_schema_version` >= 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;

CREATE TABLE `shared_boards` (
  `board_id` varchar(255) NOT NULL,
  `user_id` varchar(64) NOT NULL,
  `can_edit` tinyint(1) NOT NULL DEFAULT 0,
  `shared_at` varchar(32) NOT NULL,
  PRIMARY KEY (`board_id`,`user_id`),
  KEY `idx_shared_boards_user_id` (`user_id`),
  CONSTRAINT `fk_shared_boards_board_id_boards` FOREIGN KEY (`board_id`) REFERENCES `boards` (`board_id`) ON DELETE CASCADE,
  CONSTRAINT `fk_shared_boards_user_id_users` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_nopad_bin ROW_FORMAT=DYNAMIC;
