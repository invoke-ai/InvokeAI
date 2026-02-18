# Invoke AI Documentation

## Plan

Old Location** | **New Location** | **Action** |
|------------------|------------------|------------|
| `index.md` | `getting-started/index.mdx` | Split: intro → getting-started, features → about/features |
| `installation/quick_start.md` | `getting-started/installation.mdx` | Move + simplify |
| `installation/requirements.md` | `getting-started/system-requirements.mdx` | Move |
| `installation/manual.md` | `configuration/installation-methods/manual-install.mdx` | Move (advanced) |
| `installation/docker.md` | `configuration/installation-methods/docker.mdx` | Move (advanced) |
| `installation/models.md` | `guides/models/installing-models.mdx` | Expand + split |
| `installation/patchmatch.md` | `configuration/installation-methods/patchmatch.mdx` | Move |
| `configuration.md` | `configuration/invokeai-yaml.mdx` | Split into multiple pages |
| `faq.md` | `troubleshooting/faq.mdx` | Move + categorize |
| `help/diffusion.md` | `concepts/diffusion.mdx` | Move |
| `help/SAMPLER_CONVERGENCE.md` | `concepts/generation-parameters.mdx` | Merge |
| `help/gettingStartedWithAI.md` | `concepts/index.mdx` | Merge into concepts intro |
| `features/gallery.md` | `ui-reference/gallery.mdx` | Move |
| `features/hotkeys.md` | `ui-reference/hotkeys.mdx` | Merge with contributing/HOTKEYS.md |
| `features/low-vram.md` | `configuration/low-vram-mode.mdx` | Move |
| `features/database.md` | `development/architecture/database.mdx` | Move (dev-focused) |
| `features/orphaned_model_removal.md` | `guides/models/` | Merge into model management |
| `nodes/overview.md` | `concepts/nodes-and-workflows.mdx` + `workflows/index.mdx` | Split |
| `nodes/NODES.md` | `workflows/editor-interface.mdx` | Restructure |
| `nodes/defaultNodes.md` | `workflows/node-library.mdx` + `api-reference/node-reference/` | Split: catalog vs reference |
| `nodes/communityNodes.md` | `workflows/community-nodes.mdx` | Move |
| `nodes/contributingNodes.md` | `workflows/creating-nodes.mdx` + `development/guides/adding-nodes.mdx` | Split: user vs dev |
| `nodes/comfyToInvoke.md` | `workflows/comfyui-migration.mdx` | Move |
| `nodes/invocation-api.md` | `development/guides/api-development.mdx` | Move |
| `nodes/NODES_MIGRATION_V3_V4.md` | `workflows/` or archive | Consider archiving |
| `contributing/index.md` | `contributing/index.mdx` | Keep, simplify |
| `contributing/LOCAL_DEVELOPMENT.md` | `development/setup/local-environment.mdx` | Move |
| `contributing/dev-environment.md` | `development/setup/local-environment.mdx` | Merge |
| `contributing/ARCHITECTURE.md` | `development/architecture/overview.mdx` | Move |
| `contributing/INVOCATIONS.md` | `development/architecture/invocations.mdx` | Move |
| `contributing/MODEL_MANAGER.md` | `development/architecture/model-manager.mdx` | Move |
| `contributing/TESTS.md` | `development/guides/writing-tests.mdx` | Move |
| `contributing/NEW_MODEL_INTEGRATION.md` | `development/guides/integrating-models.mdx` | Move |
| `contributing/PR-MERGE-POLICY.md` | `development/processes/pr-merge-policy.mdx` | Move |
| `contributing/HOTKEYS.md` | `ui-reference/hotkeys.mdx` | Merge with features/hotkeys.md |
| `contributing/frontend/*` | `development/frontend/*` | Move |
| `contributing/contribution_guides/development.md` | `development/guides/` | Distribute |
| `contributing/contribution_guides/newContributorChecklist.md` | `contributing/new-contributor-guide.mdx` | Move |
| `contributing/contributors.md` | `contributing/contributors.mdx` | Keep |
| `CODE_OF_CONDUCT.md` | `contributing/code-of-conduct.mdx` | Move |
| `RELEASE.md` | `development/processes/release-process.mdx` | Move |

---

## Todo

- [ ] Write new CI docstrings generator, but to json instead of markdown.
- [ ] CI Workflow for testing md links.
