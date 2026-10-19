import type { GalleryBoard } from '@features/gallery/contracts';
import type { MainModelConfig } from '@features/generation/contracts';

import {
  canAddToCachedGalleryBoard,
  galleryBoardsOptions,
  invalidateGallery,
  invalidateGalleryBoardDestinations,
} from '@features/gallery/queries';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { queryClient } from '@platform/query/client';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryObserver } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { resolveInvocationRoute } from './invocation';
import { submitResolvedInvocation } from './invocationSubmit';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';
import { createWorkbenchStore } from './workbenchStore';

/**
 * What a submission costs in board-listing reads, pinned against the transport: a cached listing of every readable
 * board decides at once, whatever its age; only with none cached is one read awaited before the dispatch; and
 * invalidating the gallery reads back only the listings something observes.
 */

const BOARDS_LISTING_PATH = '/api/v1/boards/';

/** Answers the board listing and its count requests, holding the listing back while `holdListing` is set. */
const createBoardsServer = () => {
  const requests: string[] = [];
  let releaseListing: (() => void) | null = null;
  let holdListing = false;
  const respond = (body: unknown) =>
    new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' }, status: 200 });
  const fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(input instanceof Request ? input.url : String(input), 'http://localhost');

    requests.push(`${init?.method ?? 'GET'} ${url.pathname}${url.search}`);

    if (url.pathname === BOARDS_LISTING_PATH) {
      if (holdListing) {
        await new Promise<void>((resolve) => {
          releaseListing = resolve;
        });
      }

      return respond([]);
    }

    return respond({ total: 0 });
  };

  return {
    fetch,
    hold: () => {
      holdListing = true;
    },
    listingReads: () => requests.filter((request) => request.startsWith(`GET ${BOARDS_LISTING_PATH}`)),
    release: () => {
      holdListing = false;
      releaseListing?.();
    },
    requests,
  };
};

const board = (id: string): GalleryBoard => ({
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id,
  imageCount: 0,
  isInbox: false,
  kind: 'board',
  name: id,
  projectId: null,
  videoCount: 0,
});

const model: MainModelConfig = { base: 'sd-1', key: 'main', name: 'Main', type: 'main' };

const buildGenerateProject = () => {
  const state = workbenchReducer(createInitialWorkbenchState(), {
    type: 'setGenerateSettings',
    values: { ...getDefaultGenerateSettings(model), model, modelKey: model.key, positivePrompt: 'a cat' },
  });

  return state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
};

/** Submits a plain Generate run bound for the gallery; `dispatched` records the moment the reducer took it. */
const submitGenerate = () => {
  const project = buildGenerateProject();
  const route = resolveInvocationRoute(project, 'global', {
    destination: 'gallery',
    destinationLocked: false,
    sourceId: 'generate',
    sourceLocked: false,
  });
  const commands = createWorkbenchStore().commands;
  const dispatched = vi.spyOn(commands.generation, 'submitResolved').mockImplementation(() => undefined);

  expect(route.validationReasons).toEqual([]);

  const submission = submitResolvedInvocation({
    commands,
    formatControlLayerError: ({ code, layerName }) => `${layerName}: ${code}`,
    models: undefined,
    owner: captureAccountScope(),
    prepareCanvasInvocation: vi.fn(),
    project,
    route,
  });

  return { dispatched, submission };
};

const settle = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

let server: ReturnType<typeof createBoardsServer>;

seedArchitectureCapabilities();

beforeEach(() => {
  accountLifecycle.activate('invocation-submit-listing-cost');
  server = createBoardsServer();
  vi.stubGlobal('fetch', server.fetch);
});

afterEach(async () => {
  server.release();
  await vi.waitFor(() => expect(queryClient.isFetching()).toBe(0));
  queryClient.clear();
  vi.unstubAllGlobals();
  accountLifecycle.invalidate();
});

describe('board-listing reads before a submission', () => {
  it.each([
    ['fresh', 0],
    ['a minute old', 61_000],
  ])('dispatches at once from a cached listing of every readable board that is %s', async (_age, ageMs) => {
    queryClient.setQueryData(galleryBoardsOptions({ includeArchived: true }).queryKey, [board('own')], {
      updatedAt: Date.now() - ageMs,
    });
    server.hold();

    const { dispatched, submission } = submitGenerate();

    // Nothing was awaited: the run was dispatched with the listing still unanswered.
    await submission;
    expect(dispatched).toHaveBeenCalledTimes(1);
    expect(server.listingReads()).toHaveLength(ageMs > 0 ? 1 : 0);

    // The stale listing's one background read lands for the next run; the fresh listing is left alone.
    server.release();
    await settle();
    expect(server.listingReads()).toHaveLength(ageMs > 0 ? 1 : 0);
  });

  it('awaits exactly one listing read before the dispatch when none is cached', async () => {
    server.hold();

    const { dispatched, submission } = submitGenerate();
    await settle();

    expect(dispatched).not.toHaveBeenCalled();
    expect(server.listingReads()).toHaveLength(1);

    server.release();
    await submission;
    expect(dispatched).toHaveBeenCalledTimes(1);
    expect(server.listingReads()).toHaveLength(1);
  });
});

describe('after a board refused a run', () => {
  it('reads the listing again at once, and a submit that follows the read decides from it', async () => {
    queryClient.setQueryData(galleryBoardsOptions({ includeArchived: true }).queryKey, [board('refusing')]);

    // The server no longer lists the board; nothing observes the listing.
    await invalidateGalleryBoardDestinations(queryClient);
    expect(server.listingReads()).toHaveLength(1);
    expect(canAddToCachedGalleryBoard('refusing', queryClient)).toBe(false);

    const { dispatched, submission } = submitGenerate();
    await submission;

    expect(dispatched).toHaveBeenCalledTimes(1);
    expect(server.listingReads()).toHaveLength(1);
  });
});

describe('invalidating the gallery', () => {
  it('reads back only the board listings something observes', async () => {
    queryClient.setQueryData(galleryBoardsOptions({ includeArchived: true }).queryKey, [board('own')]);
    queryClient.setQueryData(galleryBoardsOptions().queryKey, [board('own')]);

    // No Generate, Video or paste surface observes a listing: nothing is read back.
    await invalidateGallery(queryClient);
    await settle();
    expect(server.requests).toEqual([]);

    // The panel observes its own listing of live boards, invalidated above so it is read back on mount; after that,
    // an invalidation reads back that one listing, not the one with archived boards.
    const stopObserving = new QueryObserver(queryClient, galleryBoardsOptions()).subscribe(() => undefined);
    await vi.waitFor(() => expect(queryClient.isFetching()).toBe(0));
    server.requests.length = 0;
    await invalidateGallery(queryClient);
    await vi.waitFor(() => expect(queryClient.isFetching()).toBe(0));
    stopObserving();

    expect(server.listingReads()).toEqual([expect.stringContaining('include_archived=false')]);
  });
});
