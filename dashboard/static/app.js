/* Dashboard for recorded MuJoCo states. Physics is started only by Run simulation. */
(function () {
  const live = window.WASH_MODE !== "bundle";
  const state = { run: null, index: 0, playing: false, last: 0 };

  const scenarioBadge = document.getElementById("scenario-badge");
  const runBadge = document.getElementById("run-badge");
  const statusBadge = document.getElementById("status-badge");
  const runSelect = document.getElementById("run-select");
  const jobNote = document.getElementById("job-note");
  const cards = document.getElementById("cards");
  const scrub = document.getElementById("scrub");
  const clock = document.getElementById("clock");

  function toThree(p) {
    return new THREE.Vector3(p[0], p[2], -p[1]);
  }

  function quatToThree(q) {
    const w = q[0], x = q[1], y = q[2], z = q[3];
    const R = [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ];
    const M = [[1, 0, 0], [0, 0, 1], [0, -1, 0]];
    const MT = [[1, 0, 0], [0, 0, -1], [0, 1, 0]];
    const RM = mul(R, MT);
    const Rt = mul(M, RM);
    return matrixToQuat(Rt);
  }

  function mul(a, b) {
    const c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
      }
    }
    return c;
  }

  function matrixToQuat(m) {
    const trace = m[0][0] + m[1][1] + m[2][2];
    let qw, qx, qy, qz;
    if (trace > 0) {
      const s = Math.sqrt(trace + 1) * 2;
      qw = 0.25 * s;
      qx = (m[2][1] - m[1][2]) / s;
      qy = (m[0][2] - m[2][0]) / s;
      qz = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
      const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
      qw = (m[2][1] - m[1][2]) / s;
      qx = 0.25 * s;
      qy = (m[0][1] + m[1][0]) / s;
      qz = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
      const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
      qw = (m[0][2] - m[2][0]) / s;
      qx = (m[0][1] + m[1][0]) / s;
      qy = 0.25 * s;
      qz = (m[1][2] + m[2][1]) / s;
    } else {
      const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
      qw = (m[1][0] - m[0][1]) / s;
      qx = (m[0][2] + m[2][0]) / s;
      qy = (m[1][2] + m[2][1]) / s;
      qz = 0.25 * s;
    }
    return new THREE.Quaternion(qx, qy, qz, qw).normalize();
  }

  const canvas = document.getElementById("view");
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xe7eef2, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 40);
  camera.position.set(2.3, 2.7, -4.4);
  camera.lookAt(0, 1.9, -0.7);
  scene.add(new THREE.HemisphereLight(0xf4f7fb, 0xc5ced6, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 0.85);
  sun.position.set(-2, 6, -2);
  scene.add(sun);
  const grid = new THREE.GridHelper(8, 16, 0x9aa8b5, 0xd5dee6);
  grid.position.y = 0;
  scene.add(grid);

  const glass = new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 1.8),
    new THREE.MeshStandardMaterial({ color: 0xc5dde8, transparent: true, opacity: 0.45, side: THREE.DoubleSide, roughness: 0.05 })
  );
  glass.position.set(0, 1.95, 0);
  scene.add(glass);
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x3a4048, roughness: 0.45, metalness: 0.25 });
  function bar(w, h, d, x, y, z) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), frameMat);
    mesh.position.set(x, y, z);
    scene.add(mesh);
  }
  bar(1.96, 0.07, 0.05, 0, 1.02, 0);
  bar(1.96, 0.07, 0.05, 0, 2.88, 0);
  bar(0.07, 1.96, 0.05, -0.94, 1.95, 0);
  bar(0.07, 1.96, 0.05, 0.94, 1.95, 0);
  const facade = new THREE.Mesh(
    new THREE.BoxGeometry(2.4, 2.5, 0.05),
    new THREE.MeshStandardMaterial({ color: 0xd5dbe1, roughness: 0.9 })
  );
  facade.position.set(0, 1.9, 0.08);
  scene.add(facade);

  const drone = new THREE.Group();
  const orange = new THREE.MeshStandardMaterial({ color: 0xe07a2f, roughness: 0.55 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x24282d, roughness: 0.4, metalness: 0.35 });
  const cyan = new THREE.MeshStandardMaterial({ color: 0x12838c, roughness: 0.35 });
  drone.add(new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.05, 0.12), orange));
  const L = 0.16;
  [[L, 0.03, 0], [-L, 0.03, 0], [0, 0.03, -L], [0, 0.03, L]].forEach(function (pos) {
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.008, 20), dark);
    rotor.position.set(pos[0], pos[1], pos[2]);
    drone.add(rotor);
  });
  const armX = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.02, 0.02), dark);
  armX.position.y = 0.02;
  const armZ = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.02, 0.32), dark);
  armZ.position.y = 0.02;
  drone.add(armX, armZ);
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.2, 10), cyan);
  nozzle.rotation.x = Math.PI / 2;
  nozzle.position.set(0, -0.01, 0.1);
  drone.add(nozzle);
  const nozzleTip = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 12), cyan);
  nozzleTip.position.set(0, -0.01, 0.2);
  drone.add(nozzleTip);
  scene.add(drone);

  const footprint = new THREE.Mesh(
    new THREE.CircleGeometry(0.2, 28),
    new THREE.MeshBasicMaterial({ color: 0x12838c, transparent: true, opacity: 0.35, side: THREE.DoubleSide })
  );
  footprint.position.set(0, 1.9, -0.02);
  scene.add(footprint);

  const plannedLine = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x9aa3ad })
  );
  const actualLine = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x0f6f78 })
  );
  scene.add(plannedLine, actualLine);

  function resize() {
    const width = canvas.clientWidth || 640;
    const height = canvas.clientHeight || 480;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);

  function drawHeat(grid) {
    const heat = document.getElementById("heat");
    const ctx = heat.getContext("2d");
    const rows = grid.length;
    const cols = grid[0].length;
    const cellW = heat.width / cols;
    const cellH = heat.height / rows;
    ctx.clearRect(0, 0, heat.width, heat.height);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const on = grid[r][c] > 0;
        ctx.fillStyle = on ? "#1a8f98" : "#e6eef2";
        const y = heat.height - (r + 1) * cellH;
        ctx.fillRect(c * cellW, y, cellW + 0.4, cellH + 0.4);
      }
    }
  }

  function drawTrack(run, index) {
    const track = document.getElementById("track");
    const ctx = track.getContext("2d");
    ctx.clearRect(0, 0, track.width, track.height);
    const frames = run.frames;
    const errors = frames.map(function (frame) {
      const d = [frame.p[0] - frame.ref[0], frame.p[1] - frame.ref[1], frame.p[2] - frame.ref[2]];
      return Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    });
    const maxE = Math.max(0.2, ...errors);
    ctx.strokeStyle = "#d7e0e6";
    ctx.beginPath();
    const y15 = track.height - (0.15 / maxE) * (track.height - 16) - 8;
    ctx.moveTo(8, y15);
    ctx.lineTo(track.width - 8, y15);
    ctx.stroke();
    ctx.strokeStyle = "#0f6f78";
    ctx.beginPath();
    errors.forEach(function (value, i) {
      const x = 8 + (i / Math.max(errors.length - 1, 1)) * (track.width - 16);
      const y = track.height - 8 - (value / maxE) * (track.height - 16);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const mark = 8 + (index / Math.max(frames.length - 1, 1)) * (track.width - 16);
    ctx.strokeStyle = "#9a3412";
    ctx.beginPath();
    ctx.moveTo(mark, 8);
    ctx.lineTo(mark, track.height - 8);
    ctx.stroke();
  }

  function showCards(run, frame) {
    const m = run.metrics;
    const items = [
      ["Estimated spray coverage", (100 * (frame ? frame.coverage : m.estimated_spray_coverage)).toFixed(1) + " %"],
      ["Position RMSE", m.position_rmse_m.toFixed(3) + " m"],
      ["Nozzle distance RMSE", m.nozzle_distance_rmse_m.toFixed(3) + " m"],
      ["Minimum clearance", m.min_clearance_m.toFixed(3) + " m"],
      ["Contact episodes", String(m.contact_episodes)],
      ["Max tilt", m.max_tilt_deg.toFixed(1) + " deg"],
      ["Actuator saturation", (100 * m.actuator_saturation_fraction).toFixed(1) + " %"],
      ["Completed", m.completed ? "yes" : "no"],
    ];
    if (m.recovered !== null && m.recovered !== undefined) {
      items.push(["Recovered after wind", m.recovered ? (m.recovery_time_s.toFixed(2) + " s") : "no"]);
    }
    cards.innerHTML = items.map(function (item) {
      return "<div class='card'><span>" + item[0] + "</span><b>" + item[1] + "</b></div>";
    }).join("");
  }

  function applyFrame(index) {
    const run = state.run;
    if (!run) return;
    const frame = run.frames[Math.max(0, Math.min(index, run.frames.length - 1))];
    state.index = run.frames.indexOf(frame);
    drone.position.copy(toThree(frame.p));
    drone.quaternion.copy(quatToThree(frame.q));
    const nozzle = toThree(frame.nozzle);
    footprint.position.set(nozzle.x, nozzle.y, -0.02);
    footprint.visible = frame.spraying === 1;
    const count = state.index + 1;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const point = toThree(run.frames[i].p);
      positions[i * 3] = point.x;
      positions[i * 3 + 1] = point.y;
      positions[i * 3 + 2] = point.z;
    }
    actualLine.geometry.dispose();
    actualLine.geometry = new THREE.BufferGeometry();
    actualLine.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    scrub.value = String(state.index);
    clock.textContent = frame.t.toFixed(1) + " s  " + frame.phase;
    scenarioBadge.textContent = run.scenario === "wind" ? "Wind disturbance" : "Baseline";
    runBadge.textContent = run.run_id;
    showCards(run, frame);
    drawTrack(run, state.index);
  }

  function loadRun(run) {
    state.run = run;
    state.playing = false;
    scrub.max = String(Math.max(run.frames.length - 1, 0));
    glass.position.y = run.panel.z0_m + run.panel.height_m / 2;
    const refs = new Float32Array(run.frames.length * 3);
    run.frames.forEach(function (frame, i) {
      const point = toThree(frame.ref);
      refs[i * 3] = point.x;
      refs[i * 3 + 1] = point.y;
      refs[i * 3 + 2] = point.z;
    });
    plannedLine.geometry.dispose();
    plannedLine.geometry = new THREE.BufferGeometry();
    plannedLine.geometry.setAttribute("position", new THREE.BufferAttribute(refs, 3));
    drawHeat(run.coverage_grid);
    applyFrame(0);
  }

  async function refreshList(selectId) {
    let runs;
    if (live) {
      runs = await (await fetch("/api/runs")).json();
    } else {
      runs = await (await fetch("/data/index.json")).json();
    }
    runSelect.innerHTML = "";
    runs.forEach(function (run) {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = run.scenario + "  " + run.run_id;
      runSelect.appendChild(option);
    });
    if (selectId) runSelect.value = selectId;
    if (runSelect.value) await selectListed(runSelect.value);
  }

  async function selectListed(runId) {
    const payload = live
      ? await (await fetch("/api/runs/" + runId)).json()
      : await (await fetch("/data/" + runId + ".json")).json();
    if (payload.error) {
      jobNote.textContent = payload.error;
      return;
    }
    loadRun(payload);
  }

  function setStatus(status) {
    statusBadge.textContent = status.state || "idle";
    statusBadge.className = status.state || "idle";
    if (status.error) jobNote.textContent = status.error;
    else if (status.state === "running" || status.state === "queued") {
      jobNote.textContent = (status.state === "queued" ? "Queued " : "Running ") + (status.run_id || "");
    }
  }

  async function poll() {
    if (!live) return;
    const status = await (await fetch("/api/status")).json();
    setStatus(status);
    if ((status.state === "completed" || status.state === "failed") && status.run_id && (!state.run || state.run.run_id !== status.run_id)) {
      await refreshList(status.run_id);
      jobNote.textContent = "Loaded " + status.run_id;
    }
  }

  document.getElementById("launch").addEventListener("click", async function () {
    if (!live) return;
    jobNote.textContent = "";
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: document.getElementById("scenario").value }),
    });
    const body = await response.json();
    if (!response.ok) {
      jobNote.textContent = body.error || "Could not start the simulation";
      return;
    }
    setStatus({ state: "queued", run_id: body.run_id });
  });
  runSelect.addEventListener("change", function () { selectListed(runSelect.value); });
  document.getElementById("play").addEventListener("click", function () { state.playing = true; state.last = performance.now(); });
  document.getElementById("pause").addEventListener("click", function () { state.playing = false; });
  document.getElementById("reset").addEventListener("click", function () { state.playing = false; applyFrame(0); });
  scrub.addEventListener("input", function () { state.playing = false; applyFrame(Number(scrub.value)); });

  function tick(now) {
    resize();
    if (state.playing && state.run) {
      const speed = Number(document.getElementById("speed").value);
      const dt = state.run.dt_s || 0.05;
      if (now - state.last > (dt * 1000) / speed) {
        state.last = now;
        if (state.index >= state.run.frames.length - 1) state.playing = false;
        else applyFrame(state.index + 1);
      }
    }
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }

  refreshList().catch(function (error) { jobNote.textContent = String(error); });
  if (live) setInterval(poll, 1000);
  requestAnimationFrame(tick);
})();
