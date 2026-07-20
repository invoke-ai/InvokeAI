/**
 * @deprecated Import the source-agnostic Platform registry from
 * `@platform/react/draftRegistry`. This wrapper only preserves Generation's
 * established public names while callers migrate.
 */
export {
  flushWorkbenchDrafts as flushGenerateDrafts,
  useRegisterDraftFlusher as useRegisterGenerateDraftFlusher,
} from '@platform/react/draftRegistry';
