import { api, buildV2Url } from '..';

type NodePackInfo = {
  name: string;
  path: string;
  node_count: number;
  node_types: string[];
};

type NodePackListResponse = {
  node_packs: NodePackInfo[];
  custom_nodes_path: string;
};

type InstallNodePackResponse = {
  name: string;
  success: boolean;
  message: string;
  workflows_imported: number;
  requires_dependencies: boolean;
  dependency_file: string | null;
};

type UninstallNodePackResponse = {
  name: string;
  success: boolean;
  message: string;
};

const buildCustomNodesUrl = (path: string = '') => buildV2Url(`custom_nodes/${path}`);

const customNodesApi = api.injectEndpoints({
  endpoints: (build) => ({
    listCustomNodePacks: build.query<NodePackListResponse, void>({
      query: () => ({
        url: buildCustomNodesUrl(),
        method: 'GET',
      }),
      providesTags: ['CustomNodePacks'],
    }),
    installCustomNodePack: build.mutation<InstallNodePackResponse, { source: string }>({
      query: (body) => ({
        url: buildCustomNodesUrl('install'),
        method: 'POST',
        body,
      }),
      invalidatesTags: ['CustomNodePacks', 'Workflow', 'Schema'],
    }),
    uninstallCustomNodePack: build.mutation<UninstallNodePackResponse, string>({
      query: (packName) => ({
        url: buildCustomNodesUrl(packName),
        method: 'DELETE',
      }),
      invalidatesTags: ['CustomNodePacks', 'Workflow', 'Schema'],
    }),
    reloadCustomNodes: build.mutation<{ status: string }, void>({
      query: () => ({
        url: buildCustomNodesUrl('reload'),
        method: 'POST',
      }),
      invalidatesTags: ['CustomNodePacks', 'Schema'],
    }),
  }),
});

export const {
  useListCustomNodePacksQuery,
  useInstallCustomNodePackMutation,
  useUninstallCustomNodePackMutation,
  useReloadCustomNodesMutation,
} = customNodesApi;
