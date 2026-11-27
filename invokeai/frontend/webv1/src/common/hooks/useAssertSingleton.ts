import { useEffect } from 'react';
import { assert } from 'tsafe';

const IDS = new Set<string>();

/**
 * Asserts that there is only one instance of a singleton entity. It can be a hook or a component.
 * @param id The ID of the singleton entity.
 */
export function useAssertSingleton(id: string) {
  useEffect(() => {
    assert(!IDS.has(id), `There should be only one instance of ${id}`);
    IDS.add(id);
    return () => {
      IDS.delete(id);
    };
  }, [id]);
}
