/**
 * The partial-bundle plotly distributions ship no type declarations; re-export
 * the full plotly.js types (dev dependency @types/plotly.js) for the gl2d
 * bundle, which contains only the scatter/scattergl trace modules.
 */
declare module 'plotly.js-gl2d-dist-min' {
  import * as Plotly from 'plotly.js';

  export = Plotly;
}
