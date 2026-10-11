-- TravelSATHI recommendation service — optional MySQL DDL.
--
-- This is OPTIONAL. Default setup uses data/mumbai_places.json and a local
-- JSONL feedback log — nothing here needs to exist for the service to run.
-- Use this only once the team wants the Mumbai dataset and/or feedback
-- events living in the same MySQL database as the rest of the app
-- (see DESIGN.md Section 14 for the Option A vs Option B discussion —
-- this schema implements Option A: MySQL holds structured data,
-- embeddings/vector search stay in the Python service, never in MySQL).
--
-- Run manually: mysql -u <user> -p <database> < schema.sql

CREATE TABLE IF NOT EXISTS mumbai_places (
    provider_id                  VARCHAR(64)  PRIMARY KEY,
    provider_name                VARCHAR(255) NOT NULL,
    address                      VARCHAR(500),
    latitude                     DECIMAL(9,6),
    longitude                    DECIMAL(9,6),
    category                     JSON,             -- e.g. ["shopping","markets"]
    sub_category                 VARCHAR(255),
    description                  TEXT,
    experience_tags              JSON,             -- e.g. ["local","souvenirs"]
    traveller_tags                JSON,             -- e.g. ["budget_friendly","family_friendly"]
    cost_basis                   VARCHAR(255),      -- free-text qualifier, e.g. "Free entry; spend varies"
    est_cost_per_person_inr      DECIMAL(10,2) NULL,-- NULL = unknown, NOT zero
    people_per_experience        VARCHAR(32),       -- raw string, e.g. "1-4" — parsed by the service, not MySQL
    recommended_duration         INT NULL,          -- minutes; NULL = unknown, never guessed
    opening_hours                VARCHAR(255),
    weekly_closure               VARCHAR(255) NULL, -- e.g. "Monday"; NULL = unknown, NOT "never closed"
    rating                       DECIMAL(2,1) NULL,
    review_count                 INT NULL,
    indoor_outdoor                VARCHAR(32),
    signature_items               JSON,
    local_score                  DECIMAL(3,2) NULL, -- Section 11: future crowdsourced signal, NULL until populated
    local_confidence              DECIMAL(3,2) NULL,
    local_recommendation_count    INT NULL,
    local_tips                    JSON,
    last_verified                 DATE NULL,         -- date a human last checked this row against reality
    source_url                    VARCHAR(500),
    created_at                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Feedback events (Section 22). One row per interaction; the recommendation
-- service only ever INSERTs here — analytics/training jobs read from it.
CREATE TABLE IF NOT EXISTS recommendation_feedback (
    id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id               VARCHAR(64)  NULL,       -- matches the main app's users.id; NULL for anonymous
    query_id              VARCHAR(64)  NOT NULL,   -- ties back to RecommendResponse.query_id
    place_id              VARCHAR(128) NOT NULL,   -- NormalizedPlace.id, e.g. "mumbai:TS-MUM-001"
    event                 ENUM('impression','click','save','share','booking','visit','rating','reject') NOT NULL,
    value                 DECIMAL(4,2) NULL,       -- e.g. a 1-5 star rating, when event='rating'
    dwell_time_seconds    DECIMAL(10,2) NULL,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_query_id (query_id),
    INDEX idx_place_id (place_id),
    INDEX idx_user_id (user_id)
);

-- OPTIONAL: a place_id-only cache table, matching Google's policy that only
-- the place_id itself may be stored indefinitely (see
-- retrieval/google_places.py and DESIGN.md "Google Places data policy").
-- Do NOT add columns here for name/address/rating/etc from Google — those
-- must stay in the short-lived, in-process cache in google_places.py, not
-- in a persistent table, unless you have separately confirmed your current
-- Google Maps Platform terms allow it for your use case.
CREATE TABLE IF NOT EXISTS google_place_id_cache (
    place_id     VARCHAR(128) PRIMARY KEY,
    first_seen   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used    TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
