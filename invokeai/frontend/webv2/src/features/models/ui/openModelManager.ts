import type { ModelTaxonomyType } from '@features/models/core/types';

/**
 * Opens the Model Manager for a project, on Add Models seeded with a type filter or a search when given. Hash
 * navigation works from surfaces outside the router, and the seeding module loads only when one is asked for, so the
 * manager stays out of the editor's first load. The seed is set before navigating, so the manager opens on it.
 */
export const openModelManager = ({
  modelType,
  projectId,
  search,
}: { modelType?: ModelTaxonomyType; projectId?: string | null; search?: string } = {}): void => {
  const navigate = () => {
    window.location.hash = projectId ? `#/models?project=${encodeURIComponent(projectId)}` : '#/models';
  };

  if (modelType === undefined && search === undefined) {
    navigate();
    return;
  }

  void import('@features/models/launchpad').then(({ requestAddModelsSearch, requestAddModelsTypeFilter }) => {
    if (modelType !== undefined) {
      requestAddModelsTypeFilter(modelType);
    }
    if (search !== undefined) {
      requestAddModelsSearch(search);
    }
    navigate();
  });
};
