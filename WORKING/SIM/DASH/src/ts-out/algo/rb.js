var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
import { AbrAlgorithm } from '../algo/interface';
import { ThrpPredictorGetter } from '../algo/getters';
import { Decision } from '../common/data';
import { logging } from '../common/logger';
var logger = logging('RB');
/**
 * Simple rate-based algorithm; chooses quality proportional with last segment's download
 * time bandwidth estimation.
 */
var RB = /** @class */ (function (_super) {
    __extends(RB, _super);
    function RB(video) {
        var _this = _super.call(this) || this;
        _this.bitrateArray = video.bitrateArray;
        _this.n = _this.bitrateArray.length;
        _this.bandwidth = new ThrpPredictorGetter();
        return _this;
    }
    RB.prototype.getDecision = function (metrics, index, timestamp) {
        this.bandwidth.update(metrics, this.requests);
        var bandwidth = this.bandwidth.value;
        var bitrate = 0;
        var quality = 0;
        for (var i = this.n - 1; i >= 0; i--) {
            quality = i;
            if (bandwidth >= this.bitrateArray[i]) {
                break;
            }
        }
        logger.log("bandwidth ".concat(bandwidth), "quality ".concat(quality));
        return new Decision(index, quality, timestamp);
    };
    return RB;
}(AbrAlgorithm));
export { RB };
//# sourceMappingURL=rb.js.map