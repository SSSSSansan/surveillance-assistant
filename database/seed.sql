-- Придуманные события для T5 (19 штук). Не настоящие данные MEVA.
DELETE FROM vlm_results WHERE video_id = 'test_video_01';
DELETE FROM events WHERE video_id = 'test_video_01';
DELETE FROM videos WHERE video_id = 'test_video_01';
-- Зоны не удаляются. Добавляются, если их нет; у уже существующей зоны
-- описание заполняется только когда оно пустое, чужие данные не затираются.
INSERT INTO videos (video_id, camera_id, source) VALUES ('test_video_01', 'test_cam01', 'meva');
INSERT INTO zones (camera_id, zone_id, description) VALUES ('test_cam01', 'door', 'Entrance door') ON CONFLICT (camera_id, zone_id) DO UPDATE SET description = COALESCE(zones.description, excluded.description);
INSERT INTO zones (camera_id, zone_id, description) VALUES ('test_cam01', 'desk', 'Work desk') ON CONFLICT (camera_id, zone_id) DO UPDATE SET description = COALESCE(zones.description, excluded.description);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0001', 'enter', 7, 'door', 12.4, 14.0, 0.87, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0002', 'approach_zone', 7, 'desk', 15.0, 18.5, 0.81, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0003', 'dwell_in_zone', 7, 'desk', 18.5, 35.0, 0.9, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0004', 'pick_up', 7, 'desk', 24.0, 26.0, 0.64, TRUE, 'clips/test_video_01_e0004.mp4');
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0005', 'leave_zone', 7, 'desk', 35.0, 36.2, 0.78, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0006', 'exit', 7, 'door', 41.2, 43.0, 0.85, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0007', 'enter', 8, 'door', 60.1, 61.5, 0.91, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0008', 'approach_zone', 8, 'desk', 63.0, 66.0, 0.79, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0009', 'dwell_in_zone', 8, 'desk', 66.0, 74.0, 0.88, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0010', 'put_down', 8, 'desk', 70.5, 72.0, 0.58, TRUE, 'clips/test_video_01_e0010.mp4');
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0011', 'leave_zone', 8, 'desk', 74.0, 75.2, 0.76, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0012', 'exit', 8, 'door', 78.0, 80.0, 0.83, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0013', 'abandoned_object', 8, 'desk', 95.0, 120.0, 0.71, TRUE, 'clips/test_video_01_e0013.mp4');
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0014', 'enter', 9, 'door', 150.2, 151.8, 0.89, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0015', 'exit', 9, 'door', 155.0, 156.4, 0.84, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0016', 'enter', 12, 'door', 200.0, 201.5, 0.86, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0017', 'approach_zone', 12, 'desk', 205.0, 208.0, 0.8, FALSE, NULL);
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0018', 'pick_up', 12, 'desk', 206.0, 207.5, 0.55, TRUE, 'clips/test_video_01_e0018.mp4');
INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, trigger_score, needs_vlm, clip_path) VALUES ('test_video_01', 'e0019', 'put_down', 12, 'desk', 209.0, 210.5, 0.52, TRUE, 'clips/test_video_01_e0019.mp4');
INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) VALUES ('test_video_01', 'e0004', 'A', 'confirmed', 'A person standing at the desk picks up a small bag and holds it.');
INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) VALUES ('test_video_01', 'e0010', 'A', 'confirmed', 'The person leans over the desk and places a laptop on it.');
INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) VALUES ('test_video_01', 'e0013', 'A', 'confirmed', 'A backpack stays on the desk with nobody nearby.');
INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) VALUES ('test_video_01', 'e0018', 'A', 'rejected', 'The person reaches toward the desk but no object is picked up.');
INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) VALUES ('test_video_01', 'e0004', 'B', 'confirmed', 'Person 7 in the desk zone picks up a small bag at about 24 s.');
INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) VALUES ('test_video_01', 'e0010', 'B', 'rejected', 'Person 8 is in the desk zone, but no object is visibly put down.');
INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) VALUES ('test_video_01', 'e0018', 'B', 'rejected', 'Person 12 is in the desk zone, no pick-up is visible.');
