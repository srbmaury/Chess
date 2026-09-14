import { expect, test } from 'vitest'

import styles from './styles.css?raw'


test('base form controls keep readable dark-theme colors', () => {
  expect(styles).toMatch(/select,input\{[^}]*background:[^;}]*/)
  expect(styles).toMatch(/select,input\{[^}]*color:[^;}]*/)
  expect(styles).toMatch(/select option\{[^}]*background:[^;}]*/)
  expect(styles).toMatch(/select option\{[^}]*color:[^;}]*/)
})
