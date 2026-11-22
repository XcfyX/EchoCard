-- EchoCard SQLite Schema（WAL 模式在 db.py 中开启）
-- 约定：时间戳统一为本地时间 ISO 串（YYYY-MM-DD HH:MM:SS）；日期为 YYYY-MM-DD。
-- 反偏见红线：audio_segments.audio_tags 只允许客观特征（bgm/low_volume/laughter…），
--            禁止出现“废话/无关”等主观定性标签。

CREATE TABLE IF NOT EXISTS devices (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL DEFAULT 'dji_mic_tx',  -- dji_mic_tx / lan_api / watch_dir / generic_usb
    vid           TEXT,
    pid           TEXT,
    serial        TEXT,                                -- USBSTOR 序列号（稳定标识）
    volume_label  TEXT,
    product_desc  TEXT,
    marker_root   TEXT,                                -- .echocard_source 所在设备根目录
    bound_at      TEXT NOT NULL,
    last_seen     TEXT,
    note          TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_devices_serial ON devices(serial) WHERE serial IS NOT NULL;

CREATE TABLE IF NOT EXISTS ingest_files (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id         INTEGER REFERENCES devices(id),
    source            TEXT NOT NULL DEFAULT 'usb',     -- usb / lan_api / watch_dir
    src_path          TEXT NOT NULL,
    src_sha256        TEXT NOT NULL UNIQUE,            -- 内容哈希：去重 + 完整性校验
    size_bytes        INTEGER NOT NULL,
    recorded_at       TEXT NOT NULL,                   -- 归属日期 YYYY-MM-DD
    recorded_ts       TEXT,                            -- 原始录制时间戳
    date_fallback     INTEGER NOT NULL DEFAULT 0,      -- 1 = 设备时间漂移，兜底归入当日
    raw_path          TEXT NOT NULL,                   -- 本地原始 WAV 路径
    compressed_path   TEXT,                            -- M2: Opus 压缩后路径
    status            TEXT NOT NULL DEFAULT 'imported',-- imported / processing / done / failed
    bookmark          TEXT,                            -- 硬件/人工书签标记
    starred           INTEGER NOT NULL DEFAULT 0,
    imported_at       TEXT NOT NULL,
    deleted_on_device INTEGER NOT NULL DEFAULT 0,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingest_date ON ingest_files(recorded_at);

CREATE TABLE IF NOT EXISTS photos (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id         INTEGER REFERENCES devices(id),
    source            TEXT NOT NULL,                   -- lan / watch_dir / manual
    src_sha256        TEXT NOT NULL UNIQUE,
    captured_at       TEXT NOT NULL,                   -- 归属日期 YYYY-MM-DD
    captured_ts       TEXT,                            -- EXIF/文件时间
    date_fallback     INTEGER NOT NULL DEFAULT 0,
    original_path     TEXT NOT NULL,                   -- 原图索引
    compressed_path   TEXT NOT NULL,                   -- 面向大模型的压缩图（长边≤1024，q82）
    bookmark          TEXT,
    include_in_export INTEGER NOT NULL DEFAULT 1,
    imported_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(captured_at);

-- ===== 以下表自 M2 起填充，先建好保持迁移简单 =====
CREATE TABLE IF NOT EXISTS audio_segments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id             INTEGER REFERENCES ingest_files(id) ON DELETE CASCADE,
    day                 TEXT NOT NULL,
    start_ts            TEXT NOT NULL,                 -- 挂钟起止 HH:MM:SS
    end_ts              TEXT NOT NULL,
    start_offset        REAL,
    end_offset          REAL,
    text                TEXT,
    cluster_id          INTEGER,                       -- 当日无监督聚类编号
    speaker_name        TEXT,                          -- 软匹配人名（可为 NULL）
    speaker_match_score REAL,
    asr_confidence      REAL,
    audio_tags          TEXT,                          -- JSON 数组：客观音频特征
    starred             INTEGER NOT NULL DEFAULT 0,
    segment_path        TEXT
);
CREATE INDEX IF NOT EXISTS idx_seg_day ON audio_segments(day);

CREATE TABLE IF NOT EXISTS speaker_profiles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    prototypes   TEXT NOT NULL,      -- JSON：多原型声纹向量（加权更新）
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS music_tracks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       INTEGER REFERENCES ingest_files(id) ON DELETE CASCADE,
    day           TEXT NOT NULL,
    start_ts      TEXT NOT NULL,
    end_ts        TEXT NOT NULL,
    audio_path    TEXT,
    status        TEXT NOT NULL DEFAULT 'detected',    -- detected / recognized / failed / skipped
    title         TEXT,
    artist        TEXT,
    album         TEXT,
    cover_url     TEXT,
    provider      TEXT,
    match_ts      REAL,
    confidence    REAL,
    recognized_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_music_day ON music_tracks(day);

CREATE TABLE IF NOT EXISTS daily_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    day_range   TEXT NOT NULL,       -- 'YYYY-MM-DD' 或 'YYYY-MM-DD_to_YYYY-MM-DD'
    content_md  TEXT NOT NULL,
    frontmatter TEXT,
    tags        TEXT,
    todos_json  TEXT,
    facts       TEXT,
    music       TEXT,
    source      TEXT NOT NULL DEFAULT 'reflow',  -- reflow / api
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS todos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    summary_id INTEGER REFERENCES daily_summaries(id) ON DELETE CASCADE,
    content    TEXT NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    due        TEXT,
    created_at TEXT NOT NULL
);

-- ===== FTS5 全文检索（CJK 双字分词，见 core/fts.py；ref 为不参与索引的辅助列）=====
CREATE VIRTUAL TABLE IF NOT EXISTS fts_memory USING fts5(kind UNINDEXED, day_range UNINDEXED, text, ref UNINDEXED, tokenize='unicode61');
