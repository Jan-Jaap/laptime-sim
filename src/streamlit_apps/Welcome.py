import streamlit as st


def main() -> None:
    """
    Multipage app
    Pages are in /pages folder
    """
    st.set_page_config("Laptime Simulator")
    st.markdown(
        """
            **Laptime Simulator**

            by: Jan-Jaap van de Velde

            https://github.com/Jan-Jaap/laptime-sim

            The Laptime Simulator is a tool for race line optimization. It allows you to create race tracks, select the car you will drive and the driver experience level and then simulates the race track to find the optimal line that will result in the fastest lap time.

            The optimization is done by using a form of the simulated annealing algorithm. This algorithm tries different lines and uses the fastest line found. By trying thousands of different options, we can come close (enough) to the optimum raceline.

            The simulator can also be used to compare different cars and drivers. The results of the simulation can be saved and compared with other results.

        """
    )


if __name__ == "__main__":
    main()
