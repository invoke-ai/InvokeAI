# Normalized Model Storage Migration Proposal

## Executive Summary

This proposal outlines a migration strategy from InvokeAI's current hierarchical model storage structure to a normalized, database-key-based directory structure. This change will eliminate filename conflicts, reduce database-filesystem sync issues, and provide a more robust model management system.

## Current State Analysis

### Existing Storage Structure
- **Path Format**: `models/{base}/{type}/{model_name}`
- **Example**: `models/sd-1/main/stable-diffusion-v1-5.safetensors`
- **Database**: Stores relative path in `path` field, unique constraint on (name, base, type)

### Identified Problems
1. **Filename Conflicts**: Cannot store models with identical filenames
2. **User Manipulation**: Users directly modify files, causing DB-FS sync issues
3. **Path Complexity**: Mixed handling of single files vs. directory-based models
4. **Maintenance Burden**: Complex path resolution logic scattered across codebase

## Proposed Solution

### Normalized Storage Structure
- **New Format**: `models/{model_key}/`
- **Single Files**: `models/{model_key}/model.{extension}`
- **Diffusers Models**: `models/{model_key}/{original_structure}`
- **Key Generation**: Use database primary key (UUID) as directory name

### Benefits
1. **Unique Paths**: Every model has a guaranteed unique directory
2. **Discourages Manipulation**: Obscured directory names reduce direct file access
3. **Simplified Logic**: Uniform path resolution for all model types
4. **Future-Proof**: Easily extensible for additional metadata or variants

## Implementation Strategy

### Phase 1: Foundation (Week 1-2)

#### Database Schema Changes
```sql
-- New columns for normalized storage
ALTER TABLE models ADD COLUMN normalized_path TEXT;
ALTER TABLE models ADD COLUMN is_normalized INTEGER DEFAULT 0;
CREATE INDEX normalized_path_index ON models(normalized_path);
```

#### Core Components
1. **ModelStorageCompat**: Compatibility layer for path resolution
2. **ModelMigrationService**: Handles migration of existing models
3. **Updated ModelInstallService**: New installations use normalized structure

### Phase 2: Gradual Migration (Week 3-4)

#### Migration Approach
1. **New Installations**: Automatically use normalized storage
2. **Existing Models**: Continue working with legacy paths
3. **Optional Migration**: Provide CLI tool for manual migration
4. **Batch Processing**: Migrate in chunks of 500-1000 models

#### Configuration Options
```python
class InvokeAIAppConfig(BaseSettings):
    use_normalized_storage: bool = True  # Enable for new installations
    auto_migrate_models: bool = False    # Optional auto-migration
    migration_batch_size: int = 500      # Batch size for migration
```

### Phase 3: Validation & Cleanup (Week 5)

#### Validation Steps
1. **Path Verification**: Ensure all models accessible via both paths
2. **Database Integrity**: Verify all foreign key relationships intact
3. **Performance Testing**: Benchmark model loading times
4. **Rollback Testing**: Verify recovery procedures

## Technical Implementation Details

### 1. Path Resolution Logic

```python
class ModelStorageCompat:
    def resolve_model_path(self, model: AnyModelConfig) -> Path:
        """Resolve model path supporting both storage formats."""
        models_path = self.app_config.models_path
        
        # Check normalized storage first
        if hasattr(model, 'normalized_path') and model.normalized_path:
            normalized_dir = models_path / model.normalized_path
            if normalized_dir.exists():
                # Handle single file models
                model_files = list(normalized_dir.glob("model.*"))
                if model_files:
                    return model_files[0]
                # Handle diffusers models
                return normalized_dir
        
        # Fall back to legacy path
        return models_path / model.path
```

### 2. Migration Service

```python
class ModelMigrationService:
    def migrate_model(self, key: str) -> None:
        """Migrate single model to normalized storage."""
        model = self.record_store.get_model(key)
        old_path = self.models_path / model.path
        new_path = self.models_path / key
        
        # Create normalized directory
        new_path.mkdir(parents=True, exist_ok=True)
        
        # Move model files
        if old_path.is_file():
            target = new_path / f"model{old_path.suffix}"
            shutil.move(old_path, target)
        elif old_path.is_dir():
            for item in old_path.iterdir():
                shutil.move(item, new_path / item.name)
            old_path.rmdir()
        
        # Update database
        self.record_store.update_model(key, ModelRecordChanges(
            normalized_path=key,
            is_normalized=True
        ))
```

### 3. Installation Updates

```python
def install_path(self, model_path: Path, config: ModelRecordChanges) -> str:
    """Install model using normalized storage."""
    info = self._probe(model_path, config)
    model_key = info.key or uuid_string()
    
    # Use normalized structure
    dest_path = self.app_config.models_path / model_key
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Move model to normalized location
    if model_path.is_file():
        target = dest_path / f"model{model_path.suffix}"
        move(model_path, target)
    else:
        for item in model_path.iterdir():
            move(item, dest_path / item.name)
    
    # Register with normalized path
    config.normalized_path = model_key
    config.is_normalized = True
    return self._register(dest_path, config, info)
```

## Database Migration Strategy

### Migration Safety
1. **Full Backup**: Complete database backup before migration
2. **WAL Mode**: Enable Write-Ahead Logging for concurrent access
3. **Transaction Batching**: Process 500-1000 models per transaction
4. **Progress Tracking**: Migration log table for recovery

### Migration Script Structure
```python
class MigrationNormalized:
    def migrate(self, cursor: sqlite3.Cursor) -> None:
        # 1. Schema changes
        self._add_normalized_columns(cursor)
        
        # 2. Create migration log
        self._create_migration_log(cursor)
        
        # 3. Batch migration with progress tracking
        total = self._get_model_count(cursor)
        batch_size = 500
        
        for offset in range(0, total, batch_size):
            self._migrate_batch(cursor, offset, batch_size)
            self._log_progress(cursor, offset + batch_size, total)
        
        # 4. Validation
        self._validate_migration(cursor)
        
        # 5. Cleanup
        self._optimize_indexes(cursor)
```

### Performance Considerations
1. **Index Strategy**: Add indexes before migration for batch queries
2. **Connection Pooling**: Implement appropriate timeouts
3. **Read Isolation**: Use read_uncommitted during migration
4. **Statistics Update**: Run ANALYZE after migration

## Risk Analysis & Mitigation

### Identified Risks
1. **Data Loss**: Mitigated by comprehensive backups
2. **Downtime**: Minimized through incremental migration
3. **Path Conflicts**: Prevented by UUID-based naming
4. **Performance Degradation**: Addressed by index optimization

### Rollback Strategy
1. **Database Restoration**: Restore from backup if critical failure
2. **Filesystem Recovery**: Restore model files from snapshot
3. **Partial Rollback**: Revert individual models if needed
4. **Configuration Toggle**: Disable normalized storage via config

## Testing Plan

### Unit Tests
- Path resolution for both storage formats
- Migration service operations
- Database schema changes
- Rollback procedures

### Integration Tests
- End-to-end model installation
- Mixed storage format handling
- Concurrent access during migration
- Performance benchmarks

### User Acceptance Testing
- Migration tool usability
- Performance comparison
- API compatibility verification
- UI functionality validation

## Timeline & Milestones

### Week 1-2: Foundation
- Implement database schema changes
- Create compatibility layer
- Update installation service

### Week 3-4: Migration Implementation
- Develop migration service
- Create CLI migration tool
- Implement batch processing

### Week 5: Validation & Documentation
- Comprehensive testing
- Performance optimization
- Documentation updates
- User migration guide

### Week 6: Deployment
- Staged rollout to beta users
- Monitor and address issues
- Full production deployment

## Success Metrics

1. **Zero Data Loss**: No models lost during migration
2. **Performance**: <5% degradation in model loading times
3. **Compatibility**: 100% API compatibility maintained
4. **User Impact**: <1 hour downtime for migration
5. **Adoption**: >80% of users successfully migrate within 3 months

## Conclusion

The migration to normalized model storage represents a significant improvement in InvokeAI's model management architecture. By implementing a UUID-based directory structure, we eliminate filename conflicts, reduce maintenance complexity, and provide a robust foundation for future enhancements. The phased migration approach ensures minimal disruption while maintaining full backward compatibility during the transition period.

## Appendices

### A. File Structure Examples

#### Legacy Structure
```
models/
├── sd-1/
│   ├── main/
│   │   ├── stable-diffusion-v1-5.safetensors
│   │   └── dreamshaper-v8.ckpt
│   └── lora/
│       └── style-lora.safetensors
└── sdxl/
    └── main/
        └── stable-diffusion-xl.safetensors
```

#### Normalized Structure
```
models/
├── a1b2c3d4-e5f6-7890-abcd-ef1234567890/
│   └── model.safetensors
├── b2c3d4e5-f678-9012-bcde-f23456789012/
│   └── model.ckpt
├── c3d4e5f6-7890-1234-cdef-345678901234/
│   └── model.safetensors
└── d4e5f6a7-8901-2345-def1-456789012345/
    ├── model_index.json
    ├── unet/
    ├── vae/
    └── text_encoder/
```

### B. Configuration Migration Example

```yaml
# Before Migration (config excerpt)
models:
  path: "sd-1/main/stable-diffusion-v1-5.safetensors"
  name: "Stable Diffusion v1.5"
  base: "sd-1"
  type: "main"

# After Migration (config excerpt)  
models:
  path: "sd-1/main/stable-diffusion-v1-5.safetensors"  # Legacy path preserved
  normalized_path: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  # New normalized path
  is_normalized: true
  name: "Stable Diffusion v1.5"
  base: "sd-1"
  type: "main"
```

### C. API Compatibility Matrix

| Endpoint | Legacy Support | Normalized Support | Changes Required |
|----------|---------------|-------------------|------------------|
| GET /models | ✓ | ✓ | None |
| POST /models/install | ✓ | ✓ | Internal path handling |
| DELETE /models/{key} | ✓ | ✓ | Path resolution update |
| GET /models/{key}/path | ✓ | ✓ | Compatibility layer |
| POST /models/migrate | N/A | ✓ | New endpoint |

### D. Database Schema Evolution

```sql
-- Current Schema (v7)
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    config TEXT NOT NULL,
    -- other fields...
);

-- Proposed Schema (v8)
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,           -- Legacy path (preserved)
    normalized_path TEXT,         -- New normalized path
    is_normalized INTEGER DEFAULT 0,  -- Migration status flag
    config TEXT NOT NULL,
    -- other fields...
);

-- Migration tracking table
CREATE TABLE model_migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    migration_status TEXT NOT NULL,
    migrated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (model_id) REFERENCES models(id)
);
```