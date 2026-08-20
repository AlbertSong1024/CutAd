"""cut.py 单元测试"""
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cutad import cut


# ---------------------------------------------------------------------------
# _get_duration
# ---------------------------------------------------------------------------

class TestGetDuration:
    """通过 ffprobe 读取视频时长"""

    @patch("cutad.cut.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "123.456"}})
        )
        assert cut._get_duration("v.mp4") == pytest.approx(123.456)

    @patch("cutad.cut.subprocess.run")
    def test_failure_returns_zero(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
        assert cut._get_duration("v.mp4") == 0.0

    @patch("cutad.cut.subprocess.run")
    def test_json_error_returns_zero(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="garbage"
        )
        assert cut._get_duration("v.mp4") == 0.0


# ---------------------------------------------------------------------------
# _first_keyframe_after
# ---------------------------------------------------------------------------

class TestFirstKeyframeAfter:
    @patch("cutad.cut.subprocess.run")
    def test_finds_keyframe(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0,5.0\n1,7.5\n0,8.0\n"
        )
        assert cut._first_keyframe_after("v.mp4", 6.0) == 7.5

    @patch("cutad.cut.subprocess.run")
    def test_no_keyframe_returns_t(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0,5.0\n0,8.0\n"
        )
        assert cut._first_keyframe_after("v.mp4", 6.0) == 6.0

    @patch("cutad.cut.subprocess.run")
    def test_keyframe_at_t(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1,6.0\n0,7.0\n"
        )
        assert cut._first_keyframe_after("v.mp4", 6.0) == 6.0

    @patch("cutad.cut.subprocess.run")
    def test_exception_returns_t(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("ffprobe", 60)
        assert cut._first_keyframe_after("v.mp4", 3.0) == 3.0

    @patch("cutad.cut.subprocess.run")
    def test_malformed_lines_skipped(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="garbage\nabc\n1,9.0\n"
        )
        assert cut._first_keyframe_after("v.mp4", 8.0) == 9.0


# ---------------------------------------------------------------------------
# _last_keyframe_before
# ---------------------------------------------------------------------------

class TestLastKeyframeBefore:
    @patch("cutad.cut.subprocess.run")
    def test_finds_keyframe(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1,5.0\n0,6.0\n0,7.0\n1,8.0\n0,9.0\n"
        )
        assert cut._last_keyframe_before("v.mp4", 9.5) == 8.0

    @patch("cutad.cut.subprocess.run")
    def test_no_keyframe_returns_zero(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0,5.0\n0,6.0\n"
        )
        assert cut._last_keyframe_before("v.mp4", 7.0) == 0.0

    @patch("cutad.cut.subprocess.run")
    def test_exception_returns_zero(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("ffprobe", 60)
        assert cut._last_keyframe_before("v.mp4", 5.0) == 0.0

    @patch("cutad.cut.subprocess.run")
    def test_keyframe_exactly_at_t_not_selected(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1,5.0\n1,7.0\n"
        )
        assert cut._last_keyframe_before("v.mp4", 7.0) == 5.0


# ---------------------------------------------------------------------------
# _source_video_params
# ---------------------------------------------------------------------------

class TestSourceVideoParams:
    @patch("cutad.cut.subprocess.run")
    def test_success(self, mock_run):
        def _run(cmd, **kw):
            if "-select_streams" in cmd and "v:0" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "streams": [{
                            "time_base": "1/30000",
                            "codec_name": "h264",
                            "profile": "High",
                            "level": "40",
                            "pix_fmt": "yuv420p",
                        }]
                    })
                )
            if "-select_streams" in cmd and "a:0" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "streams": [{"bit_rate": "128000"}]
                    })
                )
            return MagicMock(returncode=0, stdout="{}")

        mock_run.side_effect = _run
        params = cut._source_video_params("v.mp4")
        assert params["timescale"] == 30000
        assert params["profile"] == "high"
        assert params["level"] == "4.0"
        assert params["pix_fmt"] == "yuv420p"
        assert params["aac_bitrate"] == 128000

    @patch("cutad.cut.subprocess.run")
    def test_exception_returns_empty(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
        params = cut._source_video_params("v.mp4")
        assert params == {}

    @patch("cutad.cut.subprocess.run")
    def test_audio_probe_fails_defaults(self, mock_run):
        def _run(cmd, **kw):
            if "-select_streams" in cmd and "v:0" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "streams": [{
                            "time_base": "1/90000",
                            "codec_name": "h264",
                            "profile": "Main",
                            "level": "30",
                            "pix_fmt": "yuv420p",
                        }]
                    })
                )
            raise subprocess.CalledProcessError(1, "ffprobe")

        mock_run.side_effect = _run
        params = cut._source_video_params("v.mp4")
        assert params["aac_bitrate"] == 64000


# ---------------------------------------------------------------------------
# _count_frames
# ---------------------------------------------------------------------------

class TestCountFrames:
    @patch("cutad.cut.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="30\n"
        )
        assert cut._count_frames("v.mp4", 0, 1) == 30

    @patch("cutad.cut.subprocess.run")
    def test_zero_length_returns_zero(self, mock_run):
        assert cut._count_frames("v.mp4", 5, 5) == 0
        assert cut._count_frames("v.mp4", 10, 5) == 0
        mock_run.assert_not_called()

    @patch("cutad.cut.subprocess.run")
    def test_exception_returns_minus_one(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("ffprobe", 30)
        assert cut._count_frames("v.mp4", 0, 1) == -1

    @patch("cutad.cut.subprocess.run")
    def test_malformed_output_returns_minus_one(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc\ndef\n")
        assert cut._count_frames("v.mp4", 0, 1) == -1


# ---------------------------------------------------------------------------
# _pre_roll_frames
# ---------------------------------------------------------------------------

class TestPreRollFrames:
    @patch("cutad.cut.subprocess.run")
    def test_counts_frames_before_t(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="4.9,2\n4.95,2\n5.0,0\n"
        )
        assert cut._pre_roll_frames("v.mp4", 5.0) == 2

    @patch("cutad.cut.subprocess.run")
    def test_zero_when_first_frame_at_t(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="5.0,0\n"
        )
        assert cut._pre_roll_frames("v.mp4", 5.0) == 0

    @patch("cutad.cut.subprocess.run")
    def test_exception_returns_zero(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("ffprobe", 30)
        assert cut._pre_roll_frames("v.mp4", 5.0) == 0

    @patch("cutad.cut.subprocess.run")
    def test_malformed_lines_skipped(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="garbage\nabc,def\n4.8,0\n5.0,0\n"
        )
        assert cut._pre_roll_frames("v.mp4", 5.0) == 1


# ---------------------------------------------------------------------------
# _video_frame_count
# ---------------------------------------------------------------------------

class TestVideoFrameCount:
    @patch("cutad.cut.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="1200\n")
        assert cut._video_frame_count("v.mp4") == 1200

    @patch("cutad.cut.subprocess.run")
    def test_empty_stdout_returns_zero(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        assert cut._video_frame_count("v.mp4") == 0

    @patch("cutad.cut.subprocess.run")
    def test_exception_returns_minus_one(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
        assert cut._video_frame_count("v.mp4") == -1

    @patch("cutad.cut.subprocess.run")
    def test_malformed_returns_minus_one(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc\n")
        assert cut._video_frame_count("v.mp4") == -1


# ---------------------------------------------------------------------------
# _stream_copy_segment
# ---------------------------------------------------------------------------

class TestStreamCopySegment:
    @patch("cutad.cut._pre_roll_frames", return_value=0)
    @patch("cutad.cut._count_frames", return_value=100)
    @patch("cutad.cut._first_keyframe_after", return_value=10.0)
    @patch("cutad.cut.subprocess.run")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.exists")
    def test_keyframe_start_path(self, mock_exists, mock_stat, mock_run,
                                 mock_kf, mock_cnt, mock_pr):
        mock_exists.return_value = True
        mock_stat.return_value = MagicMock(st_size=1024)
        mock_run.return_value = MagicMock(returncode=0)

        out = Path("/tmp/seg.mp4")
        result = cut._stream_copy_segment("v.mp4", 10.0, 20.0, out)
        assert result == str(out)
        cmd = mock_run.call_args[0][0]
        assert "-frames:v" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == "100"

    @patch("cutad.cut._first_keyframe_after", return_value=12.5)
    @patch("cutad.cut.subprocess.run")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.exists")
    def test_non_keyframe_uses_t_mode(self, mock_exists, mock_stat, mock_run,
                                      mock_kf):
        mock_exists.return_value = True
        mock_stat.return_value = MagicMock(st_size=1024)
        mock_run.return_value = MagicMock(returncode=0)

        out = Path("/tmp/seg.mp4")
        result = cut._stream_copy_segment("v.mp4", 10.0, 20.0, out)
        assert result == str(out)
        cmd = mock_run.call_args[0][0]
        assert "-frames:v" not in cmd
        assert "-t" in cmd

    @patch("cutad.cut._first_keyframe_after", return_value=10.0)
    @patch("cutad.cut._count_frames", return_value=0)
    @patch("cutad.cut.subprocess.run")
    def test_zero_frames_returns_empty(self, mock_run, mock_cnt, mock_kf):
        out = Path("/tmp/seg.mp4")
        assert cut._stream_copy_segment("v.mp4", 10.0, 20.0, out) == ""
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# _cut_segment_frame_accurate
# ---------------------------------------------------------------------------

class TestCutSegmentFrameAccurate:
    @patch("cutad.cut._stream_copy_segment")
    @patch("cutad.cut._video_frame_count")
    @patch("cutad.cut._reencode_segment")
    @patch("cutad.cut._last_keyframe_before", return_value=35.0)
    @patch("cutad.cut._first_keyframe_after", return_value=10.5)
    def test_full_three_part(self, mock_kf_after, mock_kf_before,
                             mock_reenc, mock_vfc, mock_scc):
        out = Path("/tmp/seg.mp4")
        mock_reenc.return_value = "/tmp/seg.mp4"
        mock_scc.return_value = "/tmp/seg.mp4"
        mock_vfc.return_value = 100

        parts = cut._cut_segment_frame_accurate("v.mp4", 10.0, 40.0, out, {})

        assert len(parts) == 3
        assert parts[0][1]["mode"] == "reencode"
        assert parts[1][1]["mode"] == "streamcopy"
        assert parts[2][1]["mode"] == "reencode"
        assert "cut_pts" in parts[1][1]

    @patch("cutad.cut._reencode_segment")
    @patch("cutad.cut._video_frame_count")
    @patch("cutad.cut._last_keyframe_before")
    @patch("cutad.cut._first_keyframe_after", return_value=50.0)
    def test_no_keyframe_full_reencode(self, mock_kf_after, mock_kf_before,
                                       mock_vfc, mock_reenc):
        out = Path("/tmp/seg.mp4")
        mock_reenc.return_value = "/tmp/seg.mp4"
        mock_vfc.return_value = 100
        mock_kf_before.return_value = 0

        parts = cut._cut_segment_frame_accurate("v.mp4", 5, 8, out, {})
        assert len(parts) == 1
        assert parts[0][1]["mode"] == "reencode"

    @patch("cutad.cut._reencode_segment", return_value="")
    @patch("cutad.cut._stream_copy_segment", return_value="")
    @patch("cutad.cut._first_keyframe_after", return_value=10.5)
    @patch("cutad.cut._last_keyframe_before", return_value=35.0)
    def test_all_fail_returns_empty(self, mock_kf_b, mock_kf_a, mock_scc, mock_reenc):
        out = Path("/tmp/seg.mp4")
        parts = cut._cut_segment_frame_accurate("v.mp4", 10.0, 40.0, out, {})
        assert parts == []


# ---------------------------------------------------------------------------
# cut_from_ads_json
# ---------------------------------------------------------------------------

class TestCutFromAdsJson:
    @patch("cutad.cut.cut_and_join")
    @patch("builtins.open", __import__("builtins").open)
    def test_with_temp_file(self, mock_cut, tmp_path):
        ads_file = tmp_path / "ads.json"
        ads_file.write_text(json.dumps({
            "ads": [
                {"start": 30, "end": 60},
                {"start": 120, "end": 135},
            ]
        }), encoding="utf-8")
        mock_cut.return_value = "/out/no_ads.mp4"

        result = cut.cut_from_ads_json("/v.mp4", str(ads_file))
        assert result == "/out/no_ads.mp4"
        call_ads = mock_cut.call_args[0][1]
        assert len(call_ads) == 2
        assert call_ads[0] == (30, 60)
        assert call_ads[1] == (120, 135)

    @patch("cutad.cut.cut_and_join")
    def test_empty_ads_exits(self, mock_cut, tmp_path, capsys):
        ads_file = tmp_path / "ads.json"
        ads_file.write_text(json.dumps({"ads": []}), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            cut.cut_from_ads_json("/v.mp4", str(ads_file))
        assert exc.value.code == 1
        mock_cut.assert_not_called()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            cut.cut_from_ads_json("/v.mp4", "/nonexistent/ads.json")