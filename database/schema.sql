CREATE TABLE IF NOT EXISTS sensor_readings (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    side VARCHAR(32) NOT NULL DEFAULT 'left',
    co2_ppm INT NOT NULL,
    temperature_c DOUBLE NOT NULL,
    humidity_percent DOUBLE NOT NULL,
    nh3_voltage DOUBLE NOT NULL
);

CREATE TABLE IF NOT EXISTS egg_detections (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frame_id VARCHAR(64) NOT NULL DEFAULT 'base_link',
    x_m DOUBLE NOT NULL,
    y_m DOUBLE NOT NULL,
    confidence DOUBLE NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'yolo_coco_proxy'
);
