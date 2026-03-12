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
import { Decision } from '../common/data';
import { logging } from '../common/logger';
import { BufferLevelGetter } from '../algo/getters';
import { LastThrpGetter } from '../algo/getters';
import { RebufferTimeGetter } from '../algo/getters';
import { LastFetchTimeGetter } from '../algo/getters';
var logger = logging('RemoteAbr');
/**
 * RemoteAbr interface. Uses a set of derived metrics that are transmitted to a third party service,
 * which takes the Decision. The communication mecanism to the back-end of this ABR is exposed through
 * a BackendShim.
 */
var RemoteAbr = /** @class */ (function (_super) {
    __extends(RemoteAbr, _super);
    function RemoteAbr(shim) {
        var _this = _super.call(this) || this;
        _this.shim = shim;
        _this.bandwidth = new LastThrpGetter();
        _this.buffer = new BufferLevelGetter();
        _this.rebuffer = new RebufferTimeGetter();
        _this.fetch_time = new LastFetchTimeGetter();
        return _this;
    }
    RemoteAbr.prototype.getDecision = function (metrics, index, timestamp) {
        logger.log(metrics);
        this.bandwidth.update(metrics, this.requests);
        this.buffer.update(metrics);
        this.rebuffer.update(metrics);
        this.fetch_time.update(metrics, this.requests);
        // get values
        var bandwidth = this.bandwidth.value;
        var buffer = this.buffer.value;
        var rebuffer = this.rebuffer.value;
        var last_fetch_time = this.fetch_time.value;
        logger.log(bandwidth, buffer, rebuffer, last_fetch_time);
        // get decision via sync request
        var decision = undefined;
        this.shim.frontEndDecisionRequest()
            .addIndex(index)
            .addBuffer(buffer)
            .addBandwidth(bandwidth)
            .addRebuffer(rebuffer)
            .addLastFetchTime(last_fetch_time)
            .onSuccessResponse(function (res) {
            var response = JSON.parse(res.response);
            decision = response.decision;
        })
            .send();
        if (decision === undefined) {
            throw new TypeError('FrontEndDecisionRequest failed to fetch decision');
        }
        return new Decision(index, decision, timestamp);
    };
    return RemoteAbr;
}(AbrAlgorithm));
export { RemoteAbr };
//# sourceMappingURL=remote.js.map