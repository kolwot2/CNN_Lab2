"?
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1??Mb?AA??Mb?AaF#?w?N??iF#?w?N???Unknown
dHostCast"convert_image/Cast(@1?5^??@9?5^??@A?5^??@I?5^??@a?ۺ{09??i???ڡ????Unknown
bHost
DecodeJpeg"
DecodeJpeg(@1?VV??@9?VV??@A?VV??@I?VV??@a?\=?????i?\ƞ????Unknown
qHostResizeBilinear"resize/ResizeBilinear(@19??v???@99??v??s@A9??v???@I9??v??s@abN_T????i????J???Unknown
^HostMul"convert_image(@1R??X?@9R??Xr@AR??X?@IR??Xr@a?O??dݤ?i??`5}????Unknown
?HostDataset"nIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecord(@1?rh????@9?rh???c@A?rh????@I?rh???c@a?]?LD??i????J???Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(@1!?rh???@9!?rh??Z@A!?rh???@I!?rh??Z@a??$]??i6?&????Unknown
n	HostSub"per_image_standardization/sub(@1??v???@9??v??T@A??v???@I??v??T@a_?+D???i???61"???Unknown
p
HostMean"per_image_standardization/Mean(@1??|??ڳ@9??|???S@A??|??ڳ@I??|???S@a??7?_???i????|???Unknown
nHostRealDiv"per_image_standardization(@1??|?? ?@9??|?? G@A??|?? ?@I??|?? G@a?N'?H*z?i:&G۰???Unknown
?HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(@1X9?ȶ??@9X9?ȶ?A@AX9?ȶ??@IX9?ȶ?A@a???@?s?iJ????????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(@1??S?%?@9??S?%1@A??S?%?@I??S?%1@aYAA:;^c?i???8????Unknown
?HostDataset"XIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2(@1o???}x@9o???}@Ao???}x@Io???}@aB??'u?K?i???.????Unknown
[HostOneHot"one_hot(@19??v??w@99??v??@A9??v??w@I9??v??@a}`??J?i?^?&?????Unknown
?HostDataset"aIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap(@1-?????@9-????c@A/?$)\@I/?$)??a#?C?0?iOGL??????Unknown
eHost
LogicalAnd"
LogicalAnd(1?(\???G@9?(\???G@A?(\???G@I?(\???G@aދ??@?i˃$??????Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(1-??淪D@9-??淪D@A-??淪D@I-??淪D@a?&ESk?i?7?????Unknown?
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1??n? A@9??n? A@A??n? A@I??n? A@a?=???V?iF???????Unknown
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1??~j?t=@9??~j?t=@A??~j?t=@I??~j?t=@a?G?؈??iXl???????Unknown
iHostWriteSummary"WriteSummary(1?????6@9?????6@A?????6@I?????6@aٚ?۶?	?i???	????Unknown?
lHostIteratorGetNext"IteratorGetNext(1?v???1@9?v???1@A?v???1@I?v???1@a;ɉE?i?zY????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1?A`??rD@9?A`??rD@Ao????&@Io????&@a~Tr ??>i????????Unknown
dHostDataset"Iterator::Model(1j?t?K@9j?t?K@A?????K@I?????K@a?pn=??>i??ۮ????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1??v??ZG@9??v??ZG@AX9??v>@IX9??v>@aN?Lu>p?>imJK?????Unknown
eHost_Send"IteratorGetNext/_3(1m????R@9m????R@Am????R@Im????R@a??5-[d?>iO????????Unknown
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1V-???
@9V-???
@AV-???
@IV-???
@aa?J??>i??J?????Unknown
aHostIdentity"Identity(1'1?Z @9'1?Z @A'1?Z @I'1?Z @at?0j??>i¾?M?????Unknown?
eHost_Send"IteratorGetNext/_1(1P??n???9P??n???AP??n???IP??n???a?;???>i?????????Unknown*?
dHostCast"convert_image/Cast(@1?5^??@9?5^??@A?5^??@I?5^??@a???????i????????Unknown
bHost
DecodeJpeg"
DecodeJpeg(@1?VV??@9?VV??@A?VV??@I?VV??@a.K?""??i???	a????Unknown
qHostResizeBilinear"resize/ResizeBilinear(@19??v???@99??v??s@A9??v???@I9??v??s@a+?n?:??i?J?Y???Unknown
^HostMul"convert_image(@1R??X?@9R??Xr@AR??X?@IR??Xr@a?sFrȴ??i??YXV????Unknown
?HostDataset"nIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecord(@1?rh????@9?rh???c@A?rh????@I?rh???c@a?x?A????i r<?????Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(@1!?rh???@9!?rh??Z@A!?rh???@I!?rh??Z@aC??a?8??i:x?φ????Unknown
nHostSub"per_image_standardization/sub(@1??v???@9??v??T@A??v???@I??v??T@a?M&k??i??ٲGr???Unknown
pHostMean"per_image_standardization/Mean(@1??|??ڳ@9??|???S@A??|??ڳ@I??|???S@a?$?j>???iЋ/?A???Unknown
n	HostRealDiv"per_image_standardization(@1??|?? ?@9??|?? G@A??|?? ?@I??|?? G@a^??1Ku??iI???r???Unknown
?
HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(@1X9?ȶ??@9X9?ȶ?A@AX9?ȶ??@IX9?ȶ?A@a$$?????iڪS͹???Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(@1??S?%?@9??S?%1@A??S?%?@I??S?%1@a8?3C]q?iش{ه????Unknown
?HostDataset"XIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2(@1o???}x@9o???}@Ao???}x@Io???}@a5X???X?i??Ѭ????Unknown
[HostOneHot"one_hot(@19??v??w@99??v??@A9??v??w@I9??v??@a!:?X?'X?i?1?p????Unknown
?HostDataset"aIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap(@1-?????@9-????c@A/?$)\@I/?$)??a??M?o?<?iZ?^?????Unknown
eHost
LogicalAnd"
LogicalAnd(1?(\???G@9?(\???G@A?(\???G@I?(\???G@aU?,??n(?i%??M6????Unknown?
uHostFlushSummaryWriter"FlushSummaryWriter(1-??淪D@9-??淪D@A-??淪D@I-??淪D@aM?%???$?i?0V:?????Unknown?
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1??n? A@9??n? A@A??n? A@I??n? A@a?I??~V!?i]C??????Unknown
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1??~j?t=@9??~j?t=@A??~j?t=@I??~j?t=@a?*ҳ	?i?????????Unknown
iHostWriteSummary"WriteSummary(1?????6@9?????6@A?????6@I?????6@a-^?e?&?i?%&E????Unknown?
lHostIteratorGetNext"IteratorGetNext(1?v???1@9?v???1@A?v???1@I?v???1@a+?T??i??oj?????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1?A`??rD@9?A`??rD@Ao????&@Io????&@atS??U?i"|?2????Unknown
dHostDataset"Iterator::Model(1j?t?K@9j?t?K@A?????K@I?????K@a??^????>i??Y?n????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1??v??ZG@9??v??ZG@AX9??v>@IX9??v>@a;?$???>i){a??????Unknown
eHost_Send"IteratorGetNext/_3(1m????R@9m????R@Am????R@Im????R@alU?????>i??p?????Unknown
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1V-???
@9V-???
@AV-???
@IV-???
@a(???w?>i?3???????Unknown
aHostIdentity"Identity(1'1?Z @9'1?Z @A'1?Z @I'1?Z @a????ά?>ia???????Unknown?
eHost_Send"IteratorGetNext/_1(1P??n???9P??n???AP??n???IP??n???a?38͓??>i?????????Unknown2GPU