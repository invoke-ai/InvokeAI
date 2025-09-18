import { buildChatGPT4oGraph } from 'features/nodes/util/graph/generation/buildChatGPT4oGraph';
import { buildCogView4Graph } from 'features/nodes/util/graph/generation/buildCogView4Graph';
import { buildFLUXGraph } from 'features/nodes/util/graph/generation/buildFLUXGraph';
import { buildFluxKontextGraph } from 'features/nodes/util/graph/generation/buildFluxKontextGraph';
import { buildGemini2_5Graph } from 'features/nodes/util/graph/generation/buildGemini2_5Graph';
import { buildImagen3Graph } from 'features/nodes/util/graph/generation/buildImagen3Graph';
import { buildImagen4Graph } from 'features/nodes/util/graph/generation/buildImagen4Graph';
import { buildSD1Graph } from 'features/nodes/util/graph/generation/buildSD1Graph';
import { buildSD3Graph } from 'features/nodes/util/graph/generation/buildSD3Graph';
import { buildSDXLGraph } from 'features/nodes/util/graph/generation/buildSDXLGraph';
import type { GraphBuilderArg, GraphBuilderReturn } from 'features/nodes/util/graph/types';
import { assert } from 'tsafe';

type GraphBuilderFn = (arg: GraphBuilderArg) => GraphBuilderReturn | Promise<GraphBuilderReturn>;

const graphBuilderMap: Record<string, GraphBuilderFn> = {
  sdxl: buildSDXLGraph,
  'sd-1': buildSD1Graph,
  'sd-2': buildSD1Graph,
  'sd-3': buildSD3Graph,
  flux: buildFLUXGraph,
  'flux-kontext': buildFluxKontextGraph,
  cogview4: buildCogView4Graph,
  imagen3: buildImagen3Graph,
  imagen4: buildImagen4Graph,
  'chatgpt-4o': buildChatGPT4oGraph,
  'gemini-2.5': buildGemini2_5Graph,
};

export const buildGraphForBase = async (base: string, arg: GraphBuilderArg) => {
  const builder = graphBuilderMap[base];
  assert(builder, `No graph builders for base ${base}`);
  return await builder(arg);
};
