import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { describe, expect, it } from 'vitest';

import { migrateProjectDocument, serializeProjectDocumentV3 } from './projectDocument';
import { joinProjectDocument, splitProjectDocument } from './projectHalves';
import { deserializeProjectDocument } from './syncedPersistence';

type InstanceRecord = { state: { values: Record<string, unknown> }; typeId: string };

/** A schema-3 document as a build before settings layering saved it: no marker, built-in values stored as such. */
const createPreLayeringDocument = (): Record<string, unknown> => {
  const document = JSON.parse(
    JSON.stringify(serializeProjectDocumentV3(createInitialWorkbenchState().projects[0]!))
  ) as Record<string, unknown> & { widgetInstances: Record<string, InstanceRecord> };
  delete document.settingsLayering;
  document.settings = {
    antialiasProgressImages: true,
    useCpuNoise: false,
    useNsfwChecker: false,
    useWatermark: false,
    viewerAutoSwitch: 'latest',
  };
  const gallery = Object.values(document.widgetInstances).find((instance) => instance.typeId === 'gallery')!;
  gallery.state.values = {
    ...gallery.state.values,
    galleryPage: 0,
    imageDensityPercent: 50,
    showArchivedBoards: true,
    showDateBoards: false,
    thumbnailFit: 'square',
  };
  const preview = Object.values(document.widgetInstances).find((instance) => instance.typeId === 'preview')!;
  preview.state.values = { comparisonMode: 'slider', filmstripVisible: false, metadataOpen: false };
  return document;
};

const migrated = (document: Record<string, unknown>) => {
  const result = migrateProjectDocument(document);
  if (result.status !== 'migrated') {
    throw new Error(result.reason);
  }
  return result.document as Record<string, unknown> & {
    settings: Record<string, unknown>;
    widgetInstances: Record<string, InstanceRecord>;
  };
};

const valuesOf = (document: ReturnType<typeof migrated>, typeId: string): Record<string, unknown> =>
  Object.values(document.widgetInstances).find((instance) => instance.typeId === typeId)!.state.values;

describe('settings layering migration', () => {
  it('drops stored built-in values and the retired antialias flag once, and marks the document', () => {
    const document = migrated(createPreLayeringDocument());

    expect(document.settingsLayering).toBe(1);
    expect(document.settings).toEqual({ useCpuNoise: false, viewerAutoSwitch: 'latest' });

    const gallery = valuesOf(document, 'gallery');
    expect(gallery).not.toHaveProperty('showDateBoards');
    expect(gallery).not.toHaveProperty('thumbnailFit');
    expect(gallery).not.toHaveProperty('imageDensityPercent');
    // A choice that differs from the built-in is the project's pin; workspace state no setting edits stays.
    expect(gallery.showArchivedBoards).toBe(true);
    expect(gallery.galleryPage).toBe(0);
    expect(valuesOf(document, 'preview')).toEqual({ filmstripVisible: false, metadataOpen: false });
  });

  it('leaves a document that carries the marker as it is', () => {
    const document: Record<string, unknown> = { ...createPreLayeringDocument(), settingsLayering: 1 };
    const result = migrated(document);

    expect(result.settings).toBe(document.settings);
    expect(result.widgetInstances).toBe(document.widgetInstances);
  });

  it("keeps a newer build's higher marker as layered", () => {
    const document = createPreLayeringDocument() as Record<string, unknown> & {
      widgetInstances: Record<string, InstanceRecord>;
    };
    document.settingsLayering = 2;
    const gallery = Object.values(document.widgetInstances).find((instance) => instance.typeId === 'gallery')!;
    gallery.state.values = { ...gallery.state.values, showImageDimensions: false };

    const result = migrated(document);

    expect(result.settingsLayering).toBe(2);
    expect(valuesOf(result, 'gallery').showImageDimensions).toBe(false);
  });

  it('coerces a document whose marker is not a number', () => {
    const document = migrated({ ...createPreLayeringDocument(), settingsLayering: '1' });

    expect(document.settingsLayering).toBe(1);
    expect(document.settings).toEqual({ useCpuNoise: false, viewerAutoSwitch: 'latest' });
  });

  it('loads a pre-layering document with the defaults left to follow, and saves it with the marker', () => {
    const loaded = deserializeProjectDocument(createPreLayeringDocument());
    if (loaded.status !== 'loaded') {
      throw new Error(`Expected the document to load, got ${loaded.status}`);
    }

    expect(loaded.project.settings).toEqual({ useCpuNoise: false, viewerAutoSwitch: 'latest' });
    expect(serializeProjectDocumentV3(loaded.project).settingsLayering).toBe(1);
  });
});

describe('settings layering coercion reads values as their owners did', () => {
  it('drops a stored value its owner read as the built-in, and keeps a real choice', () => {
    const document = createPreLayeringDocument() as Record<string, unknown> & {
      widgetInstances: Record<string, InstanceRecord>;
    };
    const instances = Object.values(document.widgetInstances);
    const gallery = instances.find((instance) => instance.typeId === 'gallery')!;
    gallery.state.values = { boardOrderDir: 'sideways', paginationMode: 'pages', showPendingItems: 'yes' };
    const preview = instances.find((instance) => instance.typeId === 'preview')!;
    preview.state.values = { comparisonMode: 'side-by-side', filmstripVisible: 0 };
    document.settings = { useWatermark: 'on', viewerAutoSwitch: 'latest' };

    const result = migrated(document);

    // Each of these read as the built-in before layering, so they were never a choice.
    expect(valuesOf(result, 'gallery')).toEqual({});
    expect(valuesOf(result, 'preview')).toEqual({ comparisonMode: 'side-by-side' });
    expect(result.settings).toEqual({ viewerAutoSwitch: 'latest' });
  });
});

describe('settings layering across the document split', () => {
  const workspaceDefaults = () => {
    throw new Error('Both halves are present.');
  };
  const pinnedProject = () => {
    const project = createInitialWorkbenchState().projects[0]!;
    const gallery = Object.values(project.widgetInstances).find((instance) => instance.typeId === 'gallery')!;
    // Pinned at the built-in on purpose, so it holds against a user default of `aspect`.
    gallery.state = { ...gallery.state, values: { ...gallery.state.values, thumbnailFit: 'square' } };
    return project;
  };
  const roundTrip = (workspaceEdit: (workspace: Record<string, unknown>) => void = () => {}) => {
    const halves = splitProjectDocument(serializeProjectDocumentV3(pinnedProject()));
    const content = JSON.parse(halves.contentJson) as Record<string, unknown>;
    const workspace = JSON.parse(halves.workspaceJson) as Record<string, unknown>;
    workspaceEdit(workspace);
    const joined = joinProjectDocument(content, workspace, workspaceDefaults);
    if (joined.status !== 'migrated') {
      throw new Error(joined.reason);
    }
    return migrated(joined.document);
  };

  it('keeps a pin equal to the built-in: the workspace half carries the marker', () => {
    const document = roundTrip();

    expect(document.settingsLayering).toBe(1);
    expect(valuesOf(document, 'gallery').thumbnailFit).toBe('square');
  });

  it('coerces a workspace half without the marker once, then leaves it alone', () => {
    const document = roundTrip((workspace) => {
      delete workspace.settingsLayering;
    });

    expect(document.settingsLayering).toBe(1);
    expect(valuesOf(document, 'gallery')).not.toHaveProperty('thumbnailFit');
    const again = migrated(document);
    expect(again.widgetInstances).toBe(document.widgetInstances);
    expect(again.settings).toBe(document.settings);
  });
});
