/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { EdgeConnection } from './EdgeConnection';

export type Edge = {
  /**
   * The connection for the edge's from node and field
   */
  source: EdgeConnection;
  /**
   * The connection for the edge's to node and field
   */
  destination: EdgeConnection;
};

