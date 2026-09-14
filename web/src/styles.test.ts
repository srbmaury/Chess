import { readFileSync } from 'node:fs'

import { expect, test } from 'vitest'

const styles = readFileSync(new URL('./styles.css', import.meta.url), 'utf8')


test('base form controls keep readable dark-theme colors', () => {
  expect(styles).toMatch(/select,input\{[^}]*background:[^;}]*/)
  expect(styles).toMatch(/select,input\{[^}]*color:[^;}]*/)
  expect(styles).toMatch(/select option\{[^}]*background:[^;}]*/)
  expect(styles).toMatch(/select option\{[^}]*color:[^;}]*/)
})
