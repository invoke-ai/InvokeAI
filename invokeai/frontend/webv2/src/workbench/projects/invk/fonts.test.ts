import { sha256Hex } from '@platform/browser/sha256';
import { describe, expect, it, vi } from 'vitest';

import { binaryEntry, readArchive, readEntryText, textEntry, writeArchive } from './archive';
import { executeInvkExport, planInvkExport } from './exportProject';
import {
  collectFontDependencies,
  createRestoredFontLedger,
  preflightEmbeddedFonts,
  readEmbeddedFonts,
  remapFontReferences,
  restoreEmbeddedFonts,
  rollbackRestoredFonts,
  type FontArchiveTransport,
} from './fonts';
import { readInvkArchive } from './importProject';

const bytes = new Uint8Array([0, 1, 0, 0, 1, 2, 3]);
const fontRefFor = async (fontBytes: Uint8Array, id = 'original') => ({
  contentHash: await sha256Hex(fontBytes),
  family: 'Example',
  id,
  label: 'Example Regular',
});
const fontRef = (id = 'original') => fontRefFor(bytes, id);
const documentWith = (
  font: Awaited<ReturnType<typeof fontRef>>,
  secondFont: Awaited<ReturnType<typeof fontRef>> = font
) => ({
  canvas: {
    document: {
      stacks: {
        raster: [
          { id: 'one', source: { content: 'First', fontFamily: 'Example', fontRef: font, type: 'text' } },
          { id: 'two', source: { content: 'Second', fontFamily: 'Example', fontRef: secondFont, type: 'text' } },
        ],
      },
    },
  },
});
const planFor = async (includeFonts: boolean) =>
  planInvkExport({
    appVersion: '7',
    boardItems: [],
    createdAt: '2026-09-07',
    includeFonts,
    minimumCanvasSchemaVersion: 4,
    name: 'Fonts',
    projectDocument: documentWith(await fontRef()),
  });
const transportFor = (): FontArchiveTransport => ({
  download: vi.fn(() => Promise.resolve({ bytes, filename: 'Example.ttf' })),
  remove: vi.fn(async () => {}),
  upload: vi.fn(async () => ({ created: true, font: await fontRef('imported') })),
  validate: vi.fn(async () => {}),
});

describe('project font archives', () => {
  it('writes references by default without downloading private font bytes', async () => {
    const transport = await transportFor();
    let archive: Blob | undefined;
    await executeInvkExport(await planFor(false), {
      download: (blob) => {
        archive = blob;
      },
      fontTransport: transport,
    });
    const entries = await readArchive(new Uint8Array(await archive!.arrayBuffer()));
    expect(transport.download).not.toHaveBeenCalled();
    expect([...entries.keys()].some((key) => key.startsWith('fonts/'))).toBe(false);
    expect(JSON.parse(readEntryText(entries.get('manifest.json')!)).fonts).toEqual([
      {
        contentHash: await sha256Hex(bytes),
        family: 'Example',
        label: 'Example Regular',
        references: ['original'],
      },
    ]);
    const contents = await readInvkArchive(new File([archive!], 'fonts.invk'));
    expect(contents.fonts).toEqual([]);
  });

  it('embeds one file for multiple text layers, verifies it, and remaps without changing typography', async () => {
    const transport = await transportFor();
    let archive: Blob | undefined;
    await executeInvkExport(await planFor(true), {
      download: (blob) => {
        archive = blob;
      },
      fontTransport: transport,
    });
    expect(transport.download).toHaveBeenCalledTimes(1);
    const contents = await readInvkArchive(new File([archive!], 'fonts.invk'));
    expect(contents.fonts).toHaveLength(1);
    const ledger = createRestoredFontLedger();
    await preflightEmbeddedFonts(contents.fonts!, transport);
    await restoreEmbeddedFonts(contents.fonts!, ledger, transport);
    const imported = remapFontReferences(contents.projectDocument, ledger.mappings);
    expect(collectFontDependencies(imported)[0]?.references).toEqual(['imported']);
    expect(collectFontDependencies(contents.projectDocument)[0]?.references).toEqual(['original']);
    expect(JSON.stringify(imported)).toContain('First');
    expect(JSON.stringify(imported)).toContain('Second');
  });

  it('does not produce an archive when an embedded font is unavailable or changed', async () => {
    const transport = await transportFor();
    transport.download = () => Promise.resolve({ bytes: new Uint8Array([9]), filename: 'Example.ttf' });
    const download = vi.fn();
    await expect(executeInvkExport(await planFor(true), { download, fontTransport: transport })).rejects.toThrow();
    expect(download).not.toHaveBeenCalled();
  });

  it('rejects missing manifest dependencies, including an absent manifest, before accepting references-only archives', async () => {
    const first = await fontRef();
    const second = await fontRefFor(new Uint8Array([9, 8, 7]), 'second');
    const document = documentWith(first, second);
    const dependencies = collectFontDependencies(document);

    expect(dependencies).toHaveLength(2);
    await expect(readEmbeddedFonts([dependencies[0]!], document, new Map())).rejects.toThrow(
      'Font dependencies do not match the project.'
    );
    await expect(readEmbeddedFonts([], document, new Map())).rejects.toThrow(
      'Font dependencies do not match the project.'
    );
    await expect(readEmbeddedFonts(dependencies, document, new Map())).resolves.toEqual([]);
  });

  it('rejects corrupt and undeclared embedded bytes before any import mutations', async () => {
    const ref = await fontRef();
    const makeArchive = (fonts: unknown[], path: string) =>
      writeArchive(
        new Map([
          [
            'manifest.json',
            textEntry(
              JSON.stringify({
                appVersion: '7',
                contents: 'workbench-project',
                createdAt: 'now',
                fonts,
                name: 'Fonts',
                version: 2,
              })
            ),
          ],
          ['project.json', textEntry(JSON.stringify(documentWith(ref)))],
          [path, binaryEntry(new Uint8Array([9]))],
        ])
      );
    const entry = `fonts/${ref.contentHash}.ttf`;
    const dependencies = collectFontDependencies(documentWith(ref)).map((item) => ({ ...item, entry }));
    await expect(readInvkArchive(new File([await makeArchive(dependencies, entry)], 'fonts.invk'))).rejects.toThrow(
      'Invalid embedded font'
    );
    const validFontArchive = await writeArchive(
      new Map([
        [
          'manifest.json',
          textEntry(
            JSON.stringify({
              appVersion: '7',
              contents: 'workbench-project',
              createdAt: 'now',
              fonts: dependencies,
              name: 'Fonts',
              version: 2,
            })
          ),
        ],
        ['project.json', textEntry(JSON.stringify(documentWith(ref)))],
        [entry, binaryEntry(bytes)],
        ['fonts/../../private.ttf', binaryEntry(new Uint8Array([9]))],
      ])
    );
    await expect(readInvkArchive(new File([validFontArchive], 'fonts.invk'))).rejects.toThrow('not declared');
  });

  it('preflights every font before uploading and cleans up only newly created resources', async () => {
    const transport = await transportFor();
    const dependencies = collectFontDependencies(documentWith(await fontRef()));
    const embedded = [{ bytes, dependency: dependencies[0]!, filename: 'Example.ttf' }];
    transport.validate = () => Promise.reject(new Error('invalid font'));
    await expect(preflightEmbeddedFonts(embedded, transport)).rejects.toThrow('invalid font');
    expect(transport.upload).not.toHaveBeenCalled();

    const ledger = createRestoredFontLedger();
    transport.upload = async () => ({ created: false, font: await fontRef('existing') });
    await restoreEmbeddedFonts(embedded, ledger, transport);
    await rollbackRestoredFonts(ledger, transport);
    expect(transport.remove).not.toHaveBeenCalled();
    transport.upload = async () => ({ created: true, font: await fontRef('new') });
    await restoreEmbeddedFonts(embedded, ledger, transport);
    await rollbackRestoredFonts(ledger, transport);
    expect(transport.remove).toHaveBeenCalledWith('new', undefined);
  });
});
