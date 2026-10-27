let currentTargetTemp = 27.0;
let currentTripThreshold = 32.0;

const ctx = document.getElementById('multiChart').getContext('2d');
const multiChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Temperature (°C)',
                data: [],
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.04)',
                fill: true,
                borderWidth: 1.8,
                tension: 0.3,
                pointRadius: 2,
                pointHoverRadius: 4,
                yAxisID: 'y'
            },
            {
                label: 'XGBoost Predicted (t+5)',
                data: [],
                borderColor: '#818cf8',
                borderDash: [4, 4],
                borderWidth: 1.6,
                tension: 0.3,
                pointRadius: 0,
                yAxisID: 'y'
            },
            {
                label: 'Light Level (%)',
                data: [],
                borderColor: '#f59e0b',
                borderWidth: 1.5,
                tension: 0.3,
                pointRadius: 1,
                yAxisID: 'y1'
            },
            {
                label: 'Setpoint (27°C)',
                data: [],
                borderColor: '#10b981',
                borderDash: [5, 4],
                borderWidth: 1.5,
                pointRadius: 0,
                yAxisID: 'y'
            },
            {
                label: 'Trip Limit (32°C)',
                data: [],
                borderColor: '#f43f5e',
                borderDash: [3, 3],
                borderWidth: 1.2,
                pointRadius: 0,
                yAxisID: 'y'
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            mode: 'index',
            intersect: false
        },
        scales: {
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.04)' },
                ticks: {
                    color: '#64748b',
                    maxTicksLimit: 12,
                    font: { family: "'JetBrains Mono', monospace", size: 10 }
                }
            },
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: {
                    display: true,
                    text: 'Temp (°C)',
                    color: '#38bdf8',
                    font: { family: "'Inter', sans-serif", size: 11, weight: '500' }
                },
                grid: { color: 'rgba(255, 255, 255, 0.04)' },
                ticks: {
                    color: '#64748b',
                    font: { family: "'JetBrains Mono', monospace", size: 10 }
                },
                suggestedMin: 22,
                suggestedMax: 35
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: {
                    display: true,
                    text: 'Light (%)',
                    color: '#f59e0b',
                    font: { family: "'Inter', sans-serif", size: 11, weight: '500' }
                },
                grid: { drawOnChartArea: false },
                ticks: {
                    color: '#f59e0b',
                    font: { family: "'JetBrains Mono', monospace", size: 10 }
                },
                suggestedMin: 0,
                suggestedMax: 100
            }
        },
        plugins: {
            legend: {
                labels: {
                    color: '#94a3b8',
                    font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
                    boxWidth: 12,
                    boxHeight: 6,
                    padding: 12
                }
            },
            tooltip: {
                backgroundColor: '#10141f',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
                titleFont: { family: "'Inter', sans-serif", size: 11 },
                bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
                padding: 10,
                cornerRadius: 6
            }
        }
    }
});

// Submit Targets and Thresholds via API
async function submitThresholds() {
    const targetVal = parseFloat(document.getElementById('inputTargetTemp').value);
    const targetLightVal = parseFloat(document.getElementById('inputTargetLight').value);
    const tripVal = parseFloat(document.getElementById('inputTempTrip').value);
    const coolingVal = parseFloat(document.getElementById('inputAiCooling').value);
    const toast = document.getElementById('toastThreshold');

    try {
        const res = await fetch('/api/set-thresholds', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_temp: targetVal,
                target_light: targetLightVal,
                temp_trip: tripVal,
                ai_cooling: coolingVal
            })
        });
        const data = await res.json();
        if (data.status === 'success') {
            currentTargetTemp = data.thresholds.target_temp;
            currentTripThreshold = data.thresholds.temp_trip_threshold;

            multiChart.data.datasets[3].label = `Setpoint (${currentTargetTemp}°C)`;
            multiChart.data.datasets[4].label = `Trip Limit (${currentTripThreshold}°C)`;

            toast.style.display = 'inline-block';
            toast.style.background = 'rgba(16, 185, 129, 0.12)';
            toast.style.color = '#34d399';
            toast.style.border = '1px solid rgba(16, 185, 129, 0.25)';
            toast.textContent = `Applied: Setpoint ${currentTargetTemp}°C`;
            document.getElementById('badgeActiveTarget').textContent = `Target: ${currentTargetTemp}°C`;
            document.getElementById('badgeActiveTrip').textContent = `Trip: ${currentTripThreshold}°C`;
            setTimeout(() => { toast.style.display = 'none'; }, 3000);
        }
    } catch (err) {
        toast.style.display = 'inline-block';
        toast.style.background = 'rgba(244, 63, 94, 0.12)';
        toast.style.color = '#fb7185';
        toast.style.border = '1px solid rgba(244, 63, 94, 0.25)';
        toast.textContent = 'Configuration update failed';
    }
}

// ---------------- Session analytics rendering ----------------
let lastAnalytics = null;
let lastEventKey = '';

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function fmt(v, suffix = '', digits = 1) {
    return (v === null || v === undefined) ? '--' : `${Number(v).toFixed(digits)}${suffix}`;
}

function renderHeartbeat(hb, health) {
    const dot = document.getElementById('pulseDot');
    const text = document.getElementById('linkText');
    dot.className = 'pulse-dot ' + (hb.status === 'LIVE' ? 'live' : (hb.status === 'STALE' ? 'stale' : ''));

    let src = 'Awaiting Link';
    if (hb.source) {
        if (hb.source.startsWith('arduino:')) src = `Arduino ${hb.source.split(':')[1]}`;
        else if (hb.source === 'simulator') src = 'Simulator';
        else src = 'HTTP Stream';
    }
    if (hb.status === 'LIVE') {
        text.textContent = `${src} • ${hb.rate_hz ? hb.rate_hz.toFixed(2) + ' Hz' : 'Syncing'}`;
    } else if (hb.status === 'STALE') {
        text.textContent = `${src} • Idle (${Math.round(hb.last_seen_s)}s)`;
    } else {
        text.textContent = 'Awaiting Source';
    }

    [['tagS1', health.sensor_1, 'Sensor 1 (Temperature)'], ['tagS2', health.sensor_2, 'Sensor 2 (Illumination)']].forEach(([id, status, name]) => {
        const el = document.getElementById(id);
        el.className = 'sensor-tag ' + (status === 'Healthy' ? 'ok' : (status === 'No data' ? 'none' : 'bad'));
        el.title = `${name}: ${status}`;
    });
}

function renderPerformance(p) {
    document.getElementById('perfDuty').textContent = p.duty_saved_pct === null ? '--' : `${p.duty_saved_pct > 0 ? '+' : ''}${p.duty_saved_pct.toFixed(1)}%`;
    document.getElementById('perfDutySub').textContent = (p.ai_duty_pct !== null && p.shadow_duty_pct !== null)
        ? `AI ${p.ai_duty_pct.toFixed(0)}% vs. thermostat ${p.shadow_duty_pct.toFixed(0)}%`
        : 'vs. classical thermostat';

    document.getElementById('perfStability').textContent = fmt(p.stability_pct, '%');
    document.getElementById('perfStabilitySub').textContent = `within ±${p.stability_band}°C (last ${p.stability_samples})`;

    document.getElementById('perfLatency').textContent = fmt(p.latency_ms, ' ms', 2);

    const m = p.momentum_c_per_min;
    document.getElementById('perfMomentum').textContent = (m === null || Math.abs(m) > 50) ? '--' : `${m > 0 ? '+' : ''}${m.toFixed(2)}`;
}

function renderEvents(events) {
    document.getElementById('auditCount').textContent = `${events.length} event${events.length === 1 ? '' : 's'}`;
    const key = events.length ? `${events.length}|${events[0].time}|${events[0].message}` : '';
    if (key === lastEventKey) return;
    lastEventKey = key;

    document.getElementById('auditList').innerHTML = events.map(e => `
        <div class="audit-item sev-${e.severity}">
            <div class="audit-meta"><span>${e.time}</span><span class="audit-cat">${escapeHtml(e.category)}</span></div>
            <div class="audit-msg">${escapeHtml(e.message)}</div>
        </div>`).join('');
}

function renderAnalytics(a) {
    if (!a) return;
    lastAnalytics = a;
    renderHeartbeat(a.heartbeat, a.sensor_health);
    renderPerformance(a.performance);
    renderEvents(a.events);
    if (document.getElementById('summaryModal').classList.contains('open')) renderSummary(a.summary);
}

function formatDuration(s) {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return h ? `${h}h ${m}m ${sec}s` : `${m}m ${sec}s`;
}

function renderSummary(s) {
    const rows = [
        ['Session Duration', formatDuration(s.runtime_s)],
        ['Telemetry Packets', s.packets],
        ['Mean Temperature', fmt(s.avg_temp, ' °C', 2)],
        ['Mean Illumination', fmt(s.avg_light, ' %')],
        ['Setpoint Stability', fmt(s.stability_pct, ' %')],
        ['Anomaly Detections', s.anomaly_events],
        ['Emergency Trips', s.trip_events],
        ['Load Duty Avoided (vs. Thermostat)', fmt(s.duty_saved_pct, ' %')],
        ['Mean Inference Latency', fmt(s.latency_ms, ' ms', 2)]
    ];
    document.getElementById('summaryGrid').innerHTML = rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');
}

function openSummary() {
    if (lastAnalytics) renderSummary(lastAnalytics.summary);
    document.getElementById('summaryModal').classList.add('open');
}

function closeSummary() {
    document.getElementById('summaryModal').classList.remove('open');
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSummary(); });

let isInputsInitialized = false;

async function fetchStatus() {
    try {
        const res = await fetch('/api/live-status');
        if (!res.ok) return;
        const data = await res.json();

        renderAnalytics(data.analytics);

        if (data.thresholds) {
            currentTargetTemp = data.thresholds.target_temp || 27.0;
            currentTripThreshold = data.thresholds.temp_trip_threshold || 32.0;

            document.getElementById('badgeActiveTarget').textContent = `Target: ${currentTargetTemp}°C`;
            document.getElementById('badgeActiveTrip').textContent = `Trip: ${currentTripThreshold}°C`;
            document.getElementById('subtextTempTrip').textContent = `Setpoint: ${currentTargetTemp}°C | Trip: ${currentTripThreshold}°C`;
            document.getElementById('badgeLightTarget').textContent = `Target: ${data.thresholds.target_light || 50}%`;
            document.getElementById('badgePreempt').textContent = `Preempt: ${data.thresholds.ai_cooling_threshold}°C`;
            document.getElementById('subtextRecon').textContent = `Limit: ${data.thresholds.recon_threshold.toFixed(4)}`;

            multiChart.data.datasets[3].label = `Setpoint (${currentTargetTemp}°C)`;
            multiChart.data.datasets[4].label = `Trip Limit (${currentTripThreshold}°C)`;

            if (!isInputsInitialized) {
                document.getElementById('inputTargetTemp').value = currentTargetTemp;
                document.getElementById('inputTargetLight').value = data.thresholds.target_light || 50.0;
                document.getElementById('inputTempTrip').value = currentTripThreshold;
                document.getElementById('inputAiCooling').value = data.thresholds.ai_cooling_threshold;
                isInputsInitialized = true;
            }
        }

        if (!data.latest) return;
        const cur = data.latest;

        document.getElementById('valTemp').textContent = cur.temperature !== undefined ? cur.temperature.toFixed(1) + '°C' : '--.-°C';
        document.getElementById('valLight').textContent = cur.light !== undefined ? cur.light.toFixed(1) + '%' : '--.-%';

        // Target error / offset badge
        const badgeTargetOffset = document.getElementById('badgeTargetOffset');
        const tempErr = cur.temperature !== undefined ? (cur.temperature - currentTargetTemp).toFixed(1) : 0;
        if (Math.abs(tempErr) <= 0.35) {
            badgeTargetOffset.className = 'badge badge-emerald';
            badgeTargetOffset.textContent = `At Setpoint (${currentTargetTemp}°C)`;
        } else if (tempErr > 0) {
            badgeTargetOffset.className = 'badge badge-amber';
            badgeTargetOffset.textContent = `+${tempErr}°C vs Target`;
        } else {
            badgeTargetOffset.className = 'badge badge-sky';
            badgeTargetOffset.textContent = `${tempErr}°C vs Target`;
        }

        document.getElementById('subtextLight').textContent = cur.light < (data.thresholds ? data.thresholds.target_light : 50.0) 
            ? 'Below target (Aux ON)' : 'Target met (Aux OFF)';

        document.getElementById('valPredTemp').textContent = cur.predicted_temp !== undefined ? cur.predicted_temp.toFixed(1) + '°C' : '--.-°C';

        const reconLoss = cur.recon_loss !== undefined ? cur.recon_loss.toFixed(4) : '0.0000';
        document.getElementById('valReconLoss').textContent = reconLoss;

        const led1Active = cur.led_1 === 1;
        const led2Active = cur.led_2 === 1;

        document.getElementById('valLed1').textContent = led1Active ? 'ON' : 'OFF';
        document.getElementById('valLed2').textContent = led2Active ? 'ON' : 'OFF';

        document.getElementById('led1Dot').className = 'actuator-dot ' + (led1Active ? 'active-green' : '');
        document.getElementById('led2Dot').className = 'actuator-dot ' + (led2Active ? 'active-amber' : '');

        const badgeHealth = document.getElementById('badgeHealth');
        if (cur.tripped) {
            badgeHealth.textContent = 'TRIPPED';
            badgeHealth.className = 'badge badge-rose';
        } else if (cur.is_anomaly) {
            badgeHealth.textContent = 'ANOMALY';
            badgeHealth.className = 'badge badge-amber';
        } else {
            badgeHealth.textContent = 'Healthy';
            badgeHealth.className = 'badge badge-emerald';
        }

        const banner = document.getElementById('statusBanner');
        const bannerText = document.getElementById('bannerText');
        const rawMsg = cur.is_anomaly || cur.tripped
            ? `System Alert: ${cur.anomaly_reason || (cur.tripped ? 'Emergency Overheat Tripped' : 'Telemetry Anomaly')}`
            : (cur.recommendation || `System Status: Regulating toward Setpoint (${currentTargetTemp}°C)`);
        
        // Ensure clean text without legacy emojis
        const sanitizedMsg = String(rawMsg).replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '').trim();

        if (cur.is_anomaly || cur.tripped) {
            banner.className = 'banner banner-alert';
            bannerText.textContent = sanitizedMsg;
        } else {
            banner.className = 'banner banner-normal';
            bannerText.textContent = sanitizedMsg;
        }

        document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();

        if (data.history && data.history.length > 0) {
            multiChart.data.labels = data.history.map(item => item.time_str);
            multiChart.data.datasets[0].data = data.history.map(item => item.temperature);
            multiChart.data.datasets[1].data = data.history.map(item => item.predicted_temp);
            multiChart.data.datasets[2].data = data.history.map(item => item.light);
            multiChart.data.datasets[3].data = data.history.map(() => currentTargetTemp);
            multiChart.data.datasets[4].data = data.history.map(() => currentTripThreshold);
            multiChart.update();
        }

    } catch (err) {
        console.error("Polling error:", err);
    }
}

setInterval(fetchStatus, 1000);
fetchStatus();
