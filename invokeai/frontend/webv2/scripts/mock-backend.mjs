import { createServer } from 'node:http';

/**
 * Disposable in-memory backend for the browser performance measurement.
 * Serves just enough of the InvokeAI REST surface for the measured routes to
 * boot deterministically: stateful projects + client-state KV, and stable
 * empty responses for the read models (queue, models, gallery, workflows).
 * `POST /__reset` wipes all state so every sample starts from byte-identical
 * backend state. No socket.io server is provided; the app measures as
 * disconnected, which is deterministic.
 */

const createState = () => ({
  clientState: new Map(),
  nextProjectNumber: 1,
  projects: new Map(),
});

const timestamp = () => new Date().toISOString().replace('T', ' ').replace('Z', '');

const summaryOf = (project) => ({
  created_at: project.created_at,
  name: project.name,
  project_id: project.project_id,
  revision: project.revision,
  updated_at: project.updated_at,
});

const readBody = (request) =>
  new Promise((resolve, reject) => {
    const chunks = [];
    request.on('data', (chunk) => chunks.push(chunk));
    request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    request.on('error', reject);
  });

export const startMockBackend = async (port) => {
  let state = createState();

  const server = createServer(async (request, response) => {
    const url = new URL(request.url, `http://127.0.0.1:${port}`);
    const path = url.pathname;
    const method = request.method ?? 'GET';

    const json = (status, value) => {
      const body = JSON.stringify(value ?? null);
      response.writeHead(status, { 'content-type': 'application/json' });
      response.end(body);
    };

    if (method === 'POST' && path === '/__reset') {
      state = createState();
      return json(200, { ok: true });
    }

    if (path === '/api/v1/projects/' || path === '/api/v1/projects') {
      if (method === 'GET') {
        return json(200, [...state.projects.values()].map(summaryOf));
      }
      if (method === 'POST') {
        const requested = JSON.parse((await readBody(request)) || '{}');
        const now = timestamp();
        const project = {
          created_at: now,
          data: requested.data ?? {},
          name: requested.name ?? `Project Name #${state.nextProjectNumber}`,
          project_id: requested.project_id ?? `mock-project-${state.nextProjectNumber}`,
          revision: 1,
          updated_at: now,
        };
        state.nextProjectNumber += 1;
        state.projects.set(project.project_id, project);
        return json(200, project);
      }
    }

    const projectMatch = /^\/api\/v1\/projects\/([^/]+)$/.exec(path);
    if (projectMatch) {
      const projectId = decodeURIComponent(projectMatch[1]);
      const project = state.projects.get(projectId);
      if (method === 'GET') {
        return project ? json(200, project) : json(404, { detail: 'Project not found' });
      }
      if (method === 'PUT') {
        if (!project) {
          return json(404, { detail: 'Project not found' });
        }
        const requested = JSON.parse((await readBody(request)) || '{}');
        if (requested.expected_revision !== undefined && requested.expected_revision !== project.revision) {
          return json(409, { detail: 'Revision conflict' });
        }
        project.data = requested.data ?? project.data;
        project.name = requested.name ?? project.name;
        project.revision += 1;
        project.updated_at = timestamp();
        return json(200, project);
      }
      if (method === 'DELETE') {
        state.projects.delete(projectId);
        return json(200, { ok: true });
      }
    }

    if (path.startsWith('/api/v1/client_state/')) {
      const key = url.searchParams.get('key') ?? '';
      if (path.endsWith('/get_by_key')) {
        return json(200, state.clientState.get(key) ?? null);
      }
      if (path.endsWith('/set_by_key')) {
        const value = JSON.parse((await readBody(request)) || 'null');
        state.clientState.set(key, value);
        return json(200, value);
      }
      if (path.endsWith('/delete_by_key')) {
        state.clientState.delete(key);
        return json(200, { ok: true });
      }
    }

    const queueMatch = /^\/api\/v1\/queue\/[^/]+\/(.+)$/.exec(path);
    if (queueMatch) {
      const action = queueMatch[1];
      if (action === 'status') {
        return json(200, {
          processor: { is_processing: false, is_started: true },
          queue: { canceled: 0, completed: 0, failed: 0, in_progress: 0, pending: 0, queue_id: 'default', total: 0 },
        });
      }
      if (action === 'current' || action === 'next') {
        return json(200, null);
      }
      if (action === 'list_all' || action === 'items_by_ids') {
        return json(200, []);
      }
      if (action === 'list') {
        return json(200, { item_ids: [], total_count: 0 });
      }
      return json(200, null);
    }

    if (path.startsWith('/api/v2/models/stats')) {
      return json(200, null);
    }
    if (path.startsWith('/api/v2/models')) {
      return json(200, { models: [] });
    }
    if (path.startsWith('/api/v1/model_relationships')) {
      return json(200, []);
    }
    if (path.startsWith('/api/v1/app/external_providers')) {
      return json(200, []);
    }
    if (path.startsWith('/api/v1/virtual_boards/by_date')) {
      return json(200, []);
    }
    if (path.startsWith('/api/v1/images/images_by_names')) {
      return json(200, []);
    }
    if (path.startsWith('/api/v1/workflows')) {
      return json(200, { items: [], total: 0 });
    }
    if (path.startsWith('/api/v2/custom_nodes')) {
      return json(200, []);
    }

    return json(404, { detail: `No mock for ${method} ${path}` });
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });

  return {
    close: () =>
      new Promise((resolve) => {
        server.close(resolve);
        server.closeAllConnections?.();
      }),
  };
};
