# SkyAgent
SkyAgent is a small PyGame experiment focused on a single drone flying in a 2D grid world.  
It is a stripped down version of a larger multi-agent simulator, kept simple for testing ideas around movement, control, and game feel.

The game runs in fullscreen and provides a lightweight sandbox for flying, picking up parcels, and completing deliveries.

---

## Features

- Single controllable drone
- Fullscreen 2D grid world rendered with PyGame
- Smooth continuous movement with grid-based logic
- Parcel pickup and delivery mechanics
- Optional AI controller with live narration
- Timed delivery challenge mode
- Minimal HUD for feedback and debugging

---

## Installation

```bash
git clone https://github.com/opemipoVRB/SkyAgent.git
cd SkyAgent
pip install pygame-ce
python game.py
````


## Movement and Controls

SkyAgent is played in fullscreen on a grid world. You control a single drone that moves across grid cells, picks up parcels, and drops them at the delivery station located near the center of the map.

### Basic Movement

* **Move**

  * Hold the **Arrow keys** to move:

    * Up, Down, Left, Right
    * Diagonal movement is supported (for example, Up + Right)
  * Movement is continuous while keys are held
  * The drone snaps logically to grid cells but moves smoothly at high speed

* **Drone Position**

  * The left HUD displays the current grid location as:

    * `Drone cell: (col, row)`
  * The delivery station occupies a 4×4 area near the center of the map

---

### Interacting With Parcels

* **Pick Up Parcel**

  * Move the drone onto a cell containing a parcel
  * Press **Space**
  * On successful pickup:

    * The drone sprite switches to its carrying variant
    * The cell briefly flashes **green**
    * The HUD displays `carrying: True`

* **Drop Parcel**

  * Move the drone onto a delivery station cell
  * Press **Space**
  * On successful delivery:

    * The cell flashes **red**
    * The parcel is marked as delivered
    * `Total delivered` and `Session delivered` counters increase

* **Failed Pickup or Drop**

  * If the action is invalid:

    * The cell flashes **yellow**
    * No delivery is counted

* **Spawn Parcel (Mouse)**

  * Left-click on any non-station cell to spawn a parcel

---

### Timed Delivery Session

You can play freely or start a timed delivery challenge.

* **Start Timed Session**

  * Press **S**
  * Starts a 60-second delivery session
  * Resets the `Session delivered` counter
  * Remaining time is shown in the HUD as `Session time left`

* **End of Session**

  * The session ends automatically when time runs out
  * You can still move, but further deliveries are not counted

* **Reset Counters**

  * Press **R**
  * Resets:

    * Total deliveries
    * Session deliveries
    * Session timer state

---

### Controllers

The drone can be controlled manually or by an AI.

* **Switch Controller**

  * Press **TAB**
  * Toggles between:

    * `HumanAgentController`
    * `AIAgentController`
  * The active controller name is shown in the HUD

* **AI Narration Panel**

  * When AI control is active, a narration box may appear on the right
  * Displays what the AI believes it is doing or planning
  * Useful for observing and debugging AI behavior

---

### HUD and Feedback

A translucent HUD panel on the left displays:

* Active controller
* Drone grid position
* Whether the drone is carrying a parcel
* Battery level (if implemented)
* Total and session deliveries
* Remaining session time
* Number of lost drones and last known location
* Key reminders

Color feedback:

* **Green** → successful pickup
* **Red** → successful delivery
* **Yellow** → failed action

---

### Global Controls

| Action              | Key               |
| ------------------- | ----------------- |
| Move                | Arrow keys (hold) |
| Pick up / Drop      | Space             |
| Switch controller   | TAB               |
| Start timed session | S                 |
| Reset counters      | R                 |
| Quit                | ESC               |

---

### Startup Flow

* The game starts with a splash screen

  * Press any key or mouse button to continue
* A setup screen follows:

  * Choose how many parcels to spawn
  * Use **Up / Down** or type digits directly
  * Press **Enter** to start
  * Press **ESC** to quit

---

## Notes

SkyAgent is intentionally small and modular. It is designed as a playable sandbox rather than a polished game or research framework, making it easy to extend, refactor, or experiment with new mechanics.

---