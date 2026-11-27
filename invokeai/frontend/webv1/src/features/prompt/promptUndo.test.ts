import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { clearPromptUndo, consumePromptUndo, setPromptUndo } from './promptUndo';

describe('promptUndo', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    clearPromptUndo();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('set / consume flow', () => {
    it('should return the saved prompt when consumed', () => {
      setPromptUndo('hello world');
      expect(consumePromptUndo()).toBe('hello world');
    });

    it('should return null on second consume (single-use)', () => {
      setPromptUndo('hello');
      consumePromptUndo();
      expect(consumePromptUndo()).toBeNull();
    });

    it('should return the latest prompt when set multiple times', () => {
      setPromptUndo('first');
      setPromptUndo('second');
      expect(consumePromptUndo()).toBe('second');
    });
  });

  describe('timeout expiry', () => {
    it('should return null after 30 seconds', () => {
      setPromptUndo('will expire');
      vi.advanceTimersByTime(30_001);
      expect(consumePromptUndo()).toBeNull();
    });

    it('should still be available just before 30 seconds', () => {
      setPromptUndo('still valid');
      vi.advanceTimersByTime(29_999);
      expect(consumePromptUndo()).toBe('still valid');
    });

    it('should reset the timer when set is called again', () => {
      setPromptUndo('first');
      vi.advanceTimersByTime(20_000);
      setPromptUndo('second');
      vi.advanceTimersByTime(20_000);
      // 40s total, but only 20s since the last set
      expect(consumePromptUndo()).toBe('second');
    });
  });

  describe('clear behavior', () => {
    it('should make consume return null', () => {
      setPromptUndo('cleared');
      clearPromptUndo();
      expect(consumePromptUndo()).toBeNull();
    });

    it('should not throw when clearing with nothing set', () => {
      expect(() => clearPromptUndo()).not.toThrow();
    });

    it('should cancel the pending timeout', () => {
      setPromptUndo('will be cleared');
      clearPromptUndo();
      // Even if we wait, nothing bad happens
      vi.advanceTimersByTime(60_000);
      expect(consumePromptUndo()).toBeNull();
    });
  });

  describe('edge cases', () => {
    it('should return null when nothing has been set', () => {
      expect(consumePromptUndo()).toBeNull();
    });

    it('should handle empty string as a valid prompt', () => {
      setPromptUndo('');
      expect(consumePromptUndo()).toBe('');
    });
  });
});
