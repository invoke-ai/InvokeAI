import type { GalleryItemsFilter } from '@features/gallery/queries';
import type { ParseDynamicPromptsRequest, ParseDynamicPromptsResponse } from '@features/generation/prompts';
import type {
  WorkflowAsyncGeneratorRequest,
  WorkflowBatchItem,
  WorkflowGeneratorResolutions,
  WorkflowPendingGenerator,
} from '@features/workflow/core/batch';
import type { QueryClient } from '@tanstack/react-query';

import { galleryItemNamesOptions } from '@features/gallery/queries';
import { dynamicPromptsQueryOptions } from '@features/generation/prompts';
import { queryOptions } from '@tanstack/react-query';

export interface WorkflowGeneratorQueryResult {
  items: WorkflowBatchItem[];
  /** The backend's own complaint about the request (bad dynamic-prompt syntax), shown instead of the items. */
  error: string | null;
}

const toDynamicPromptsRequest = (
  request: Extract<WorkflowAsyncGeneratorRequest, { kind: 'dynamicPrompts' }>
): ParseDynamicPromptsRequest => ({
  combinatorial: request.combinatorial,
  max_prompts: request.maxPrompts,
  prompt: request.prompt,
  seed: request.combinatorial ? null : request.seed,
});

const toBoardFilter = (
  request: Extract<WorkflowAsyncGeneratorRequest, { kind: 'boardImages' }>
): GalleryItemsFilter => ({
  boardId: request.boardId,
  galleryView: request.category,
  searchTerm: '',
});

const fromDynamicPrompts = (response: ParseDynamicPromptsResponse): WorkflowGeneratorQueryResult => ({
  error: response.error ?? null,
  items: response.prompts,
});

const fromBoardImages = (page: { items: readonly { kind: string; name: string }[] }): WorkflowGeneratorQueryResult => ({
  error: null,
  items: page.items.filter((item) => item.kind === 'image').map((item) => ({ image_name: item.name })),
});

/**
 * The query behind an async generator: dynamic prompts or a board's images. The preview widget and the submit
 * path share the underlying queries so a preview that already loaded needs no second round trip.
 */
export const getWorkflowGeneratorQueryOptions = (request: WorkflowAsyncGeneratorRequest) =>
  request.kind === 'dynamicPrompts'
    ? queryOptions({ ...dynamicPromptsQueryOptions(toDynamicPromptsRequest(request)), select: fromDynamicPrompts })
    : queryOptions({ ...galleryItemNamesOptions(toBoardFilter(request)), select: fromBoardImages });

/** A seed for an unseeded random prompt draw; the server needs an integer and a fresh one per submission. */
const drawSeed = (): number => Math.floor(Math.random() * 0x7fffffff);

const fetchWorkflowGenerator = (
  queryClient: QueryClient,
  request: WorkflowAsyncGeneratorRequest
): Promise<WorkflowGeneratorQueryResult> =>
  request.kind === 'dynamicPrompts'
    ? queryClient.fetchQuery(dynamicPromptsQueryOptions(toDynamicPromptsRequest(request))).then(fromDynamicPrompts)
    : // Fetched fresh so the batch sees the board as it is now, not as the picker last showed it.
      queryClient
        .fetchQuery({ ...galleryItemNamesOptions(toBoardFilter(request)), staleTime: 0 })
        .then(fromBoardImages);

/** Resolves every pending async generator for a submission; unseeded random prompts draw a new seed each time. */
export const resolveWorkflowGenerators = async (
  queryClient: QueryClient,
  pending: readonly WorkflowPendingGenerator[]
): Promise<{ resolutions: WorkflowGeneratorResolutions; errors: string[] }> => {
  const resolutions: WorkflowGeneratorResolutions = {};
  const errors: string[] = [];

  await Promise.all(
    pending.map(async ({ key, nodeId, nodeLabel, request }) => {
      const effectiveRequest: WorkflowAsyncGeneratorRequest =
        request.kind === 'dynamicPrompts' && !request.combinatorial && request.seed === null
          ? { ...request, seed: drawSeed() }
          : request;

      try {
        const result = await fetchWorkflowGenerator(queryClient, effectiveRequest);

        if (result.error) {
          errors.push(`${nodeLabel}: ${result.error}`);
        } else {
          resolutions[nodeId] = { items: result.items, key };
        }
      } catch (error) {
        errors.push(`${nodeLabel}: ${error instanceof Error ? error.message : 'the request failed.'}`);
      }
    })
  );

  return { errors, resolutions };
};
