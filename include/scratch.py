
class SamplingPolicy:

    def __init__(self, samples: list=None,
                 speed: float=0.1,
                 visualize: bool=False,
                 number: int=None,
                 name: str=None):

        """
        Parameters
        ----------
        samples : list, optional
            List of samples. The default is None.
        speed : float, optional
            Speed of the agent. The default is 0.1.
        visualize : bool, optional
            Visualize the policy. The default is False.
        number : int, optional
            Number of the figure. The default is None.
        name : str, optional
            Name of the policy. The default is None.
        """

        self._name = name if name is not None else "SamplingPolicy"
        self._samples = samples
        if samples is None:
            self._samples = [np.array([-speed/np.sqrt(2),
                                       speed/np.sqrt(2)]),
                             np.array([0., speed]),
                             np.array([speed/np.sqrt(2),
                                       speed/np.sqrt(2)]),
                             np.array([-speed, 0.]),
                             np.array([0., 0.]),
                             np.array([speed, 0.]),
                             np.array([-speed/np.sqrt(2),
                                       -speed/np.sqrt(2)]),
                             np.array([0., -speed]),
                             np.array([speed/np.sqrt(2),
                                       -speed/np.sqrt(2)])]
            logger(f"{self.__class__} using default samples [2D movements]")

        # np.random.shuffle(self._samples)

        self._num_samples = len(self._samples)
        self._samples_indexes = list(range(self._num_samples))

        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

        # render
        self._number = number
        self.visualize = visualize
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 6))
            logger(f"%visualizing {self.__class__}")

    def __len__(self):
        return self._num_samples

    def __str__(self):

        return f"{self._name}(#samples={self._num_samples})"

    def __call__(self, keep: bool=False) -> tuple:

        # --- keep the current velocity
        if keep and self._idx is not None:
            return self._velocity.copy(), False, self._idx

        # --- first sample
        if self._idx is None:
            self._idx = np.random.choice(
                            self._samples_indexes, p=self._p)
            self._available_idxs.remove(self._idx)
            self._velocity = self._samples[self._idx]
            return self._velocity.copy(), False, self._idx, self._values

        # --- all samples have been tried
        if len(self._available_idxs) == 0:

            # self._idx = np.random.choice(self._num_samples,
            #                                   p=self._p)

            if np.where(self._values == 0)[0].size > 1:
                self._idx = np.random.choice(
                                np.where(self._values == 0)[0])
            else:
                self._idx = np.argmax(self._values)

            self._velocity = self._samples[self._idx]
            # print(f"{self._name} || selected: {self._idx} | " + \
            #     f"{self._values.max()} | values: {np.around(self._values, 2)} v={np.around(self._velocity*1000, 2)}")
            return self._velocity.copy(), True, self._idx, self._values

        # --- sample again
        p = self._p[self._available_idxs].copy()
        p /= p.sum()
        self._idx = np.random.choice(
                        self._available_idxs,
                        p=p)
        self._available_idxs.remove(self._idx)
        self._velocity = self._samples[self._idx]

        return self._velocity.copy(), False, self._idx, self._values

    def update(self, score: float):

        # --- normalize the score
        # score = pcnn.generalized_sigmoid(x=score,
        #                                  alpha=-0.5,
        #                                  beta=1.)

        self._values[self._idx] = score

        # --- update the probability
        # a raw score of 0. becomes 0.5 [sigmoid]
        # and this ends in a multiplier of 1. [id]
        # self._p[self._idx] *= (0.5 + score)

        # normalize
        # self._p = self._p / self._p.sum()

    def get_state(self):

        return {"values": self._values,
                "idx": self._idx,
                "p": self._p,
                "velocity": self._velocity,
                "available_idxs": self._available_idxs}

    def set_state(self, state: dict):

        self._values = state["values"]
        self._idx = state["idx"]
        self._p = state["p"]
        self._velocity = state["velocity"]
        self._available_idxs = state["available_idxs"]

    def render(self, values: np.ndarray=None,
               action_values: np.ndarray=None):

        if not self.visualize:
            return

        # self._values = (self._values.max() - self._values) / \
        #     (self._values.max() - self._values.min())
        # self._values = np.where(np.isnan(self._values), 0,
        #                         self._values)

        self.ax.clear()

        if action_values is not None:
            self.ax.imshow(action_values.reshape(3, 3),
                           cmap="RdBu_r", vmin=-1.1, vmax=1.1,
                           aspect="equal",
                           interpolation="nearest")
        else:
            self.ax.imshow(self._values.reshape(3, 3),
                           cmap="RdBu_r", vmin=-3.1, vmax=3.1,
                           aspect="equal",
                           interpolation="nearest")

        # labels inside each square
        for i in range(3):
            for j in range(3):

                if values is not None:
                    text = "".join([f"{np.around(v, 2)}\n" for v in values[3*i+j]])
                else:
                    text = f"{self._samples[3*i+j][1]:.3f}\n" + \
                          f"{self._samples[3*i+j][0]:.3f}"
                self.ax.text(j, i, f"{text}",
                             ha="center", va="center",
                             color="black",
                             fontsize=13)

        # self.ax.bar(range(self._num_samples), self._values)
        # self.ax.set_xticks(range(self._num_samples))
        # self.ax.set_xticklabels(["stay", "up", "right",
        #                          "down", "left"])
        # self.ax.set_xticklabels(np.around(self._values, 2))
        self.ax.set_xlabel("Action")
        self.ax.set_title(f"Action Space")
        self.ax.set_yticks(range(3))
        # self.ax.set_ylim(-1, 1)
        self.ax.set_xticks(range(3))

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/fig{self._number}.png")
            return

        self.fig.canvas.draw()

    def reset(self):
        self._idx = None
        self._available_idxs = list(range(self._num_samples))
        self._p = np.ones(self._num_samples) / self._num_samples
        self._velocity = self._samples[0]
        self._values = np.zeros(self._num_samples)

    def has_collided(self):

        self._velocity = -self._velocity






class LeakyVariable:

    def __init__(self, eq: float=0., tau: float=10,
                 name: str="leaky", ndim: int=1,
                 visualize: bool=False,
                 number: int=None,
                 max_record: int=100):

        """
        Parameters
        ----------
        eq : float
            Default 0.
        tau : float
            Default 10
        threshold : float
            Default 0.
        """

        self.name = name
        self.eq = np.array([eq]*ndim).reshape(-1, 1) if ndim > 1 else np.array([eq])
        self.ndim = ndim
        self._v = np.ones(1)*self.eq if ndim == 1 else np.ones((ndim, 1))*self.eq
        self.tau = tau
        self.record = []
        self._max_record = max_record
        self._visualize = False
        self._number = number

        # figure configs
        if self._visualize:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))

    def __repr__(self):
        return f"{self.name}(eq={self.eq}, tau={self.tau})"

    def __call__(self, x: float=0., eq: np.ndarray=None,
                 simulate: bool=False):

        if simulate:
            if eq is not None:
                self.eq = eq
            return self._v + (self.eq - self._v) / self.tau + x

        if eq is not None:
            self.eq = eq
        self._v += (self.eq - self._v) / self.tau + x
        self._v = np.maximum(0., self._v)
        # self.v = np.clip(self.v, -1, 1.)
        self.record += [self._v.tolist()]
        if len(self.record) > self._max_record:
            del self.record[0]

        return self._v

    def reset(self):
        self._v = self.eq
        self.record = []

    def render(self):

        if not self._visualize:
            return

        self.ax.clear()
        self.ax.plot(range(len(self.record)), self.record)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid()
        self.ax.set_title(f"{self.name} |" +
            f" v={np.around(self._v, 2).tolist()}")

        if self._number is not None:
            self.fig.savefig(f"{FIGPATH}/{self._number}.png")
            return
        self.fig.canvas.draw()

