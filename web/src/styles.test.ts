import { expect, test } from 'vitest'

import styles from './styles.css?inline'


test('base form controls keep readable dark-theme colors', () => {
  expect(styles).toContain('select,input{background:#11130f;color:#f3f4ef')
  expect(styles).toContain('select option{background:#11130f;color:#f3f4ef')
})
