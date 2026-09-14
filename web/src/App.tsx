import { useState } from 'react'

import CommunityControls from './CommunityControls'
import CoachApp from './LegacyApp'

export default function App() {
  const [profileRevision, setProfileRevision] = useState(0)
  const [activeProfile, setActiveProfile] = useState<string | null | undefined>(undefined)

  return <>
    <CommunityControls
      onProfileChanged={() => setProfileRevision((value) => value + 1)}
      onActiveProfileChanged={setActiveProfile}
    />
    {activeProfile ? <CoachApp key={`${activeProfile}-${profileRevision}`} /> : null}
  </>
}
