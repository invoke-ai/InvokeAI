/**
 * Adapted from https://github.com/mourner/simplify-js/
 *
 * Copyright (c) 2017, Vladimir Agafonkin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *    1. Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright notice, this list
 *       of conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

import type { Coordinate } from 'features/controlLayers/store/types';
import { assert } from 'tsafe';

// square distance between 2 points
function getSqDist(p1: Coordinate, p2: Coordinate) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;

  return dx * dx + dy * dy;
}

// square distance from a point to a segment
function getSqSegDist(p: Coordinate, p1: Coordinate, p2: Coordinate) {
  let x = p1.x;
  let y = p1.y;
  let dx = p2.x - x;
  let dy = p2.y - y;

  if (dx !== 0 || dy !== 0) {
    const t = ((p.x - x) * dx + (p.y - y) * dy) / (dx * dx + dy * dy);

    if (t > 1) {
      x = p2.x;
      y = p2.y;
    } else if (t > 0) {
      x += dx * t;
      y += dy * t;
    }
  }

  dx = p.x - x;
  dy = p.y - y;

  return dx * dx + dy * dy;
}
// rest of the code doesn't care about point format

// basic distance-based simplification
function simplifyRadialDist(points: Coordinate[], sqTolerance: number): Coordinate[] {
  let prevPoint = points[0]!;
  const newPoints = [prevPoint];
  let point: Coordinate;

  for (let i = 1, len = points.length; i < len; i++) {
    point = points[i]!;

    if (getSqDist(point, prevPoint!) > sqTolerance) {
      newPoints.push(point);
      prevPoint = point;
    }
  }

  if (prevPoint !== point!) {
    newPoints.push(point!);
  }

  return newPoints;
}

function simplifyDPStep(
  points: Coordinate[],
  first: number,
  last: number,
  sqTolerance: number,
  simplified: Coordinate[]
) {
  let maxSqDist = sqTolerance;
  let index;

  for (let i = first + 1; i < last; i++) {
    const sqDist = getSqSegDist(points[i]!, points[first]!, points[last]!);

    if (sqDist > maxSqDist) {
      index = i;
      maxSqDist = sqDist;
    }
  }

  if (maxSqDist > sqTolerance) {
    if (index! - first > 1) {
      simplifyDPStep(points, first, index!, sqTolerance, simplified);
    }
    simplified.push(points[index!]!);
    if (last - index! > 1) {
      simplifyDPStep(points, index!, last, sqTolerance, simplified);
    }
  }
}

// simplification using Ramer-Douglas-Peucker algorithm
function simplifyDouglasPeucker(points: Coordinate[], sqTolerance: number) {
  const last = points.length - 1;

  const simplified = [points[0]!];
  simplifyDPStep(points, 0, last, sqTolerance, simplified);
  simplified.push(points[last]!);

  return simplified;
}

type SimplifyOptions = {
  tolerance?: number;
  highestQuality?: boolean;
};

// both algorithms combined for awesome performance
function simplifyCoords(points: Coordinate[], options?: SimplifyOptions): Coordinate[] {
  const { tolerance, highestQuality } = { ...options, tolerance: 1, highestQuality: false };

  if (points.length <= 2) {
    return points;
  }

  const sqTolerance = tolerance * tolerance;

  const firstPassPoints = highestQuality ? points : simplifyRadialDist(points, sqTolerance);
  const secondPassPoints = simplifyDouglasPeucker(firstPassPoints, sqTolerance);

  return secondPassPoints;
}

function coordsToFlatNumbersArray(coords: Coordinate[]): number[] {
  return coords.flatMap((coord) => [coord.x, coord.y]);
}

function flatNumbersArrayToCoords(array: number[]): Coordinate[] {
  assert(array.length % 2 === 0, 'Array length must be even');
  const coords: Coordinate[] = [];
  for (let i = 0; i < array.length; i += 2) {
    coords.push({ x: array[i]!, y: array[i + 1]! });
  }
  return coords;
}

export function simplifyFlatNumbersArray(array: number[], options?: SimplifyOptions): number[] {
  return coordsToFlatNumbersArray(simplifyCoords(flatNumbersArrayToCoords(array), options));
}
