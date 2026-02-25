import { GetAlgorithm } from '../algo/selector';
import { logging, exportLogs } from '../common/logger';
import { StatsTracker } from '../component/stats';
import { SetQualityController, onEvent } from '../component/abr';
import { Interceptor } from '../component/intercept';
import { QualityController } from '../controller/quality';
import { StatsController } from '../controller/stats';
var logger = logging('App');
/**
 * Front-End based ABR implementations.
 *
 * Uses an algorithm from the `src/algo` folder.
 */
var FrontEndApp = /** @class */ (function () {
    function FrontEndApp(player, recordMetrics, shim, name, videoInfo) {
        this.tracker = new StatsTracker(player);
        this.interceptor = new Interceptor(videoInfo);
        this.shim = shim;
        this.statsController = new StatsController();
        this.qualityController = new QualityController();
        this.algorithm = GetAlgorithm(name, shim, videoInfo);
        this.algorithmName = name;
        this.recordMetrics = recordMetrics;
        this.max_index = videoInfo.info[videoInfo.bitrateArray[0]].length;
        SetQualityController(this.qualityController);
    }
    FrontEndApp.prototype.start = function () {
        var _this = this;
        logger.log("Starting App.");
        // When the QualityController is queried by the ABR module, send the metrics to the 
        // experimental setup for centralization. If the underlying algorithm is Minerva, send the 
        // metrics to the backend as well so that the backend modifies the congestion control 
        // accordingly.
        this.qualityController
            .onGetQuality(function (index) {
            _this.tracker.getMetrics();
            var controller = _this.qualityController;
            var metrics = _this.statsController.metrics;
            var timestamp = (metrics.playerTime.slice(-1)[0] || { 'timestamp': 0 }).timestamp;
            _this.statsController.advance(timestamp);
            if (_this.recordMetrics) {
                _this.shim
                    .metricsLoggingRequest()
                    .addStats(metrics.serialize(true))
                    .send();
            }
            if (_this.algorithmName === "minerva") {
                _this.shim
                    .metricsRequest()
                    .addStats(metrics.serialize(true))
                    .send();
            }
            var decision = _this.algorithm.getDecision(metrics, index, timestamp);
            controller.addPiece(decision);
        });
        // When the stream finishes, send the full logs to the experimental setup.
        var eos = function (_unused) {
            logger.log('End of stream');
            if (_this.recordMetrics) {
                var logs = exportLogs();
                _this.shim
                    .metricsLoggingRequest()
                    .addLogs(logs)
                    .addComplete()
                    .send();
            }
        };
        onEvent("endOfStream", function (context) { return eos(context); });
        onEvent("PLAYBACK_ENDED", function (context) { return eos(context); });
        onEvent("Detected unintended removal", function (context) {
            logger.log('Detected unintdended removal!');
            var controller = context.scheduleController;
            controller.start();
        });
        // Setup the interceptor with full bypasses. Once each request is made, send it to 
        // the algorithm so that it can process the download times. Also update the tracker with the 
        // new metrics and signal the QualityController to advance.
        this.interceptor
            .onRequest(function (ctx, index) {
            if (index == _this.max_index) {
                // Finish stream if we downloaded everything
                eos(ctx);
                return;
            }
            _this.algorithm.newRequest(ctx);
            logger.log('Advance', index + 1);
            _this.qualityController.advance(index + 1);
            _this.tracker.getMetrics();
        })
            .start();
        this.tracker.registerCallback(function (metrics) {
            _this.statsController.addMetrics(metrics);
        });
        this.tracker.start();
    };
    return FrontEndApp;
}());
export { FrontEndApp };
//# sourceMappingURL=front_end.js.map