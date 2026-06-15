import { ApiError, getApiErrorMessage } from '@workbench/backend/http';
import { describe, expect, it } from 'vitest';

import {
  createProfileSchema,
  createSetupSchema,
  createUserFormSchema,
  getPasswordStrength,
  loginSchema,
} from './schemas';

describe('password strength', () => {
  it('mirrors the backend rules', () => {
    expect(getPasswordStrength('short')).toBe('weak');
    expect(getPasswordStrength('alllowercase')).toBe('moderate');
    expect(getPasswordStrength('NoDigitsHere')).toBe('moderate');
    expect(getPasswordStrength('Str0ngEnough')).toBe('strong');
  });
});

describe('login schema', () => {
  it('requires a valid email and a password', () => {
    expect(loginSchema.safeParse({ email: 'not-an-email', password: 'x', rememberMe: false }).success).toBe(false);
    expect(loginSchema.safeParse({ email: 'a@b.com', password: '', rememberMe: false }).success).toBe(false);
    expect(loginSchema.safeParse({ email: 'a@b.com', password: 'pw', rememberMe: true }).success).toBe(true);
  });

  it('accepts special-use domains, matching the backend', () => {
    expect(loginSchema.safeParse({ email: 'admin@localhost', password: 'pw', rememberMe: false }).success).toBe(true);
    expect(loginSchema.safeParse({ email: 'dev@invoke.local', password: 'pw', rememberMe: false }).success).toBe(true);
  });
});

describe('setup schema', () => {
  it('enforces strength only in strict mode', () => {
    const base = { confirmPassword: 'simple', displayName: '', email: 'a@b.com', password: 'simple' };

    expect(createSetupSchema(false).safeParse(base).success).toBe(true);
    expect(createSetupSchema(true).safeParse(base).success).toBe(false);
    expect(
      createSetupSchema(true).safeParse({ ...base, confirmPassword: 'Str0ngEnough', password: 'Str0ngEnough' }).success
    ).toBe(true);
  });

  it('rejects mismatched confirmation', () => {
    const result = createSetupSchema(false).safeParse({
      confirmPassword: 'other',
      displayName: '',
      email: 'a@b.com',
      password: 'simple',
    });

    expect(result.success).toBe(false);
  });
});

describe('profile schema', () => {
  it('treats an empty new password as no change', () => {
    const result = createProfileSchema(true).safeParse({
      confirmPassword: '',
      currentPassword: '',
      displayName: 'Me',
      newPassword: '',
    });

    expect(result.success).toBe(true);
  });

  it('requires the current password and confirmation when changing', () => {
    const schema = createProfileSchema(false);

    expect(
      schema.safeParse({ confirmPassword: 'newpw', currentPassword: '', displayName: '', newPassword: 'newpw' }).success
    ).toBe(false);
    expect(
      schema.safeParse({ confirmPassword: 'nope', currentPassword: 'old', displayName: '', newPassword: 'newpw' })
        .success
    ).toBe(false);
    expect(
      schema.safeParse({ confirmPassword: 'newpw', currentPassword: 'old', displayName: '', newPassword: 'newpw' })
        .success
    ).toBe(true);
  });
});

describe('user form schema', () => {
  it('requires email and password when creating', () => {
    const schema = createUserFormSchema(true, true);

    expect(schema.safeParse({ displayName: '', email: '', isAdmin: false, password: 'Str0ngEnough' }).success).toBe(
      false
    );
    expect(schema.safeParse({ displayName: '', email: 'a@b.com', isAdmin: false, password: 'weak' }).success).toBe(
      false
    );
    expect(
      schema.safeParse({ displayName: '', email: 'a@b.com', isAdmin: true, password: 'Str0ngEnough' }).success
    ).toBe(true);
  });

  it('allows an empty password when editing', () => {
    const schema = createUserFormSchema(true, false);

    expect(schema.safeParse({ displayName: 'New Name', email: '', isAdmin: false, password: '' }).success).toBe(true);
    expect(schema.safeParse({ displayName: '', email: '', isAdmin: false, password: 'weak' }).success).toBe(false);
  });
});

describe('getApiErrorMessage', () => {
  it('unwraps FastAPI detail strings', () => {
    expect(getApiErrorMessage(new ApiError('{"detail":"Incorrect email or password"}', 401), 'fallback')).toBe(
      'Incorrect email or password'
    );
  });

  it('unwraps the first validation issue', () => {
    const body = JSON.stringify({ detail: [{ loc: ['body', 'email'], msg: 'value is not a valid email address' }] });

    expect(getApiErrorMessage(new ApiError(body, 422), 'fallback')).toBe('value is not a valid email address');
  });

  it('falls back for non-JSON bodies and unknown errors', () => {
    expect(getApiErrorMessage(new ApiError('', 500), 'fallback')).toBe('fallback');
    expect(getApiErrorMessage(new ApiError('plain text', 500), 'fallback')).toBe('plain text');
    expect(getApiErrorMessage('nope', 'fallback')).toBe('fallback');
  });
});
